package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// Stress Test Configuration
// ============================================================================

type TestConfig struct {
	URL            string
	Concurrency    int
	Requests       int
	TimeoutSec     int
	PayloadFile    string
}

var (
	url            string
	concurrency    int
	requests       int
	timeoutSec     int
	payloadFile    string
)

func init() {
	flag.StringVar(&url, "url", "http://127.0.0.1:18081/v1/chat/completions", "Target URL")
	flag.IntVar(&concurrency, "c", 10, "Concurrent requests")
	flag.IntVar(&requests, "n", 100, "Total requests")
	flag.IntVar(&timeoutSec, "timeout", 60, "Request timeout in seconds")
	flag.StringVar(&payloadFile, "payload", "test_payload.json", "JSON payload file")
	flag.Parse()
}

// ============================================================================
// Test Results
// ============================================================================

type Result struct {
	OK       bool
	Status   int
	Duration time.Duration
	Error    string
}

type Summary struct {
	Total       int
	Success     int
	Failed      int
	MinLatency  time.Duration
	MaxLatency  time.Duration
	AvgLatency  time.Duration
	Throughput  float64
	Latencies   []time.Duration
}

func (s *Summary) Add(r Result) {
	s.Total++
	if r.OK {
		s.Success++
	} else {
		s.Failed++
	}
	s.Latencies = append(s.Latencies, r.Duration)

	if r.Duration < s.MinLatency || s.MinLatency == 0 {
		s.MinLatency = r.Duration
	}
	if r.Duration > s.MaxLatency {
		s.MaxLatency = r.Duration
	}
}

func (s *Summary) Calculate() {
	if s.Total > 0 {
		var totalLatency time.Duration
		for _, l := range s.Latencies {
			totalLatency += l
		}
		s.AvgLatency = totalLatency / time.Duration(s.Total)
	}
	if s.Latencies[len(s.Latencies)-1] > 0 {
		s.Throughput = float64(s.Success) / s.Latencies[len(s.Latencies)-1].Seconds()
	}
}

// ============================================================================
// HTTP Client
// ============================================================================

var client = &http.Client{
	Timeout: time.Duration(timeoutSec) * time.Second,
}

func sendRequest(url string, payload []byte) Result {
	start := time.Now()

	req, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return Result{OK: false, Error: err.Error(), Duration: time.Since(start)}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return Result{OK: false, Error: err.Error(), Duration: time.Since(start)}
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	duration := time.Since(start)

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return Result{OK: true, Status: resp.StatusCode, Duration: duration}
	}

	return Result{OK: false, Status: resp.StatusCode, Duration: duration, Error: string(body)}
}

// ============================================================================
// Load Payload
// ============================================================================

func loadPayload(filename string) ([]byte, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		// Return default payload if file doesn't exist
		payload := map[string]interface{}{
			"model": "Qwen/Qwen3.5-27B",
			"messages": []map[string]string{
				{"role": "user", "content": "Hello, how are you?"},
			},
			"max_tokens": 50,
		}
		return json.Marshal(payload)
	}
	return data, nil
}

// ============================================================================
// Worker
// ============================================================================

func worker(id int, wg *sync.WaitGroup, payload []byte, results chan Result) {
	defer wg.Done()

	for {
		select {
		case <-time.After(time.Duration(id) * 10 * time.Millisecond):
			// Stagger startup
		default:
		}

		result := sendRequest(url, payload)
		results <- result
		break
	}
}

// ============================================================================
// Main
// ============================================================================

func main() {
	fmt.Println("=== Codex Bridge Stress Test ===")
	fmt.Printf("Target: %s\n", url)
	fmt.Printf("Concurrency: %d\n", concurrency)
	fmt.Printf("Total Requests: %d\n", requests)
	fmt.Println()

	// Load payload
	payload, err := loadPayload(payloadFile)
	if err != nil {
		fmt.Printf("Error loading payload: %v\n", err)
		os.Exit(1)
	}

	var payloadMap map[string]interface{}
	if err := json.Unmarshal(payload, &payloadMap); err != nil {
		fmt.Printf("Error parsing payload: %v\n", err)
		os.Exit(1)
	}

	// Ensure stream is false for testing
	payloadMap["stream"] = false
	payload, _ = json.Marshal(payloadMap)

	fmt.Printf("Payload: %s\n", string(payload))
	fmt.Println()

	// Run test
	startTime := time.Now()
	results := make(chan Result, requests)
	var wg sync.WaitGroup

	// Warmup
	fmt.Print("Warmup... ")
	for i := 0; i < 3; i++ {
		sendRequest(url, payload)
	}
	fmt.Println("done")

	// Run concurrent requests
	fmt.Printf("Running %d requests with %d concurrent workers...\n", requests, concurrency)
	atomic.StoreInt32(&running, 1)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go worker(i, &wg, payload, results)
	}

	// Wait for all requests to complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	summary := &Summary{}
	for result := range results {
		summary.Add(result)
		if !result.OK {
			fmt.Printf("Request failed: %s\n", result.Error)
		}
	}

	totalTime := time.Since(startTime)
	summary.Calculate()

	// Print results
	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("Total Requests:  %d\n", summary.Total)
	fmt.Printf("Success:         %d (%.1f%%)\n", summary.Success, float64(summary.Success)/float64(summary.Total)*100)
	fmt.Printf("Failed:          %d\n", summary.Failed)
	fmt.Printf("Total Time:      %v\n", totalTime)
	fmt.Printf("Min Latency:     %v\n", summary.MinLatency)
	fmt.Printf("Max Latency:     %v\n", summary.MaxLatency)
	fmt.Printf("Avg Latency:     %v\n", summary.AvgLatency)
	fmt.Printf("Throughput:      %.2f req/s\n", summary.Throughput)

	// Percentiles
	if len(summary.Latencies) > 0 {
		sorted := make([]time.Duration, len(summary.Latencies))
		copy(sorted, summary.Latencies)
		// Simple sort - not most efficient but works for small samples
		for i := 0; i < len(sorted)-1; i++ {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[i] > sorted[j] {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}

		fmt.Println()
		fmt.Println("=== Percentiles ===")
		fmt.Printf("P50:  %v\n", sorted[len(sorted)*50/100])
		fmt.Printf("P90:  %v\n", sorted[len(sorted)*90/100])
		fmt.Printf("P95:  %v\n", sorted[len(sorted)*95/100])
		fmt.Printf("P99:  %v\n", sorted[len(sorted)*99/100])
	}

	if summary.Failed > 0 {
		os.Exit(1)
	}
}

var running int32