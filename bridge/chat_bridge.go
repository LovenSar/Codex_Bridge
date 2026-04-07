package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// Config
type Config struct {
	Host            string
	Port            int
	UpstreamBaseURL string
	UpstreamAPIKey  string
	DefaultModel    string
	TimeoutSec      int
}

var (
	httpAddr        string
	httpPort        int
	upstreamBaseURL string
	upstreamAPIKey  string
	defaultModel    string
	timeoutSec      int
	requestCounter  int64
)

func init() {
	flag.StringVar(&httpAddr, "host", "127.0.0.1", "Bridge listening host")
	flag.IntVar(&httpPort, "port", 18081, "Bridge listening port")
	flag.StringVar(&upstreamBaseURL, "upstream-base-url", getEnv("CODEX_UPSTREAM_BASE_URL", "http://100.122.242.51:8000/v1"), "Upstream LLM API base URL")
	flag.StringVar(&upstreamAPIKey, "upstream-api-key", getEnv("CODEX_UPSTREAM_API_KEY", "dummy"), "Upstream API key")
	flag.StringVar(&defaultModel, "default-model", getEnv("CODEX_MODEL", "Qwen/Qwen3.5-27B"), "Default model name")
	flag.IntVar(&timeoutSec, "timeout", 120, "Request timeout in seconds")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func loadConfig() *Config {
	flag.Parse()
	return &Config{
		Host:            httpAddr,
		Port:            httpPort,
		UpstreamBaseURL: strings.TrimSuffix(upstreamBaseURL, "/"),
		UpstreamAPIKey:  upstreamAPIKey,
		DefaultModel:    defaultModel,
		TimeoutSec:      timeoutSec,
	}
}

// Global shared client
var sharedClient *http.Client

func initClient() {
	sharedClient = &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        200,
			MaxIdleConnsPerHost: 200,
			IdleConnTimeout:     90 * time.Second,
		},
		Timeout: time.Duration(120) * time.Second,
	}
}

type Handler struct {
	config       *Config
	responsesURL string
	modelsURL    string
}

func NewHandler(cfg *Config) *Handler {
	base := strings.TrimSuffix(cfg.UpstreamBaseURL, "/")
	var baseV1 string
	if strings.HasSuffix(base, "/v1") {
		baseV1 = base
	} else {
		baseV1 = base + "/v1"
	}

	return &Handler{
		config:       cfg,
		responsesURL: baseV1 + "/responses",
		modelsURL:    baseV1 + "/models",
	}
}

func (h *Handler) sendJSON(w http.ResponseWriter, status int, payload interface{}) {
	data, _ := json.Marshal(payload)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(data)
}

func (h *Handler) proxyRequest(method, urlStr string, body []byte) (int, []byte, error) {
	req, err := http.NewRequest(method, urlStr, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+h.config.UpstreamAPIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := sharedClient.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, respBody, nil
}

func normalizeRole(role string) string {
	role = strings.ToLower(strings.TrimSpace(role))
	if role == "developer" {
		return "system"
	}
	if role == "system" || role == "user" || role == "assistant" {
		return role
	}
	return "user"
}

func coalesceSystemMessages(input []map[string]interface{}) []map[string]interface{} {
	var systemMsgs []map[string]interface{}
	var otherMsgs []map[string]interface{}

	for _, m := range input {
		if r, ok := m["role"].(string); ok && r == "system" {
			systemMsgs = append(systemMsgs, m)
		} else {
			otherMsgs = append(otherMsgs, m)
		}
	}

	if len(systemMsgs) == 0 {
		return input
	}

	var mergedContent string
	for i, msg := range systemMsgs {
		if c, ok := msg["content"].(string); ok {
			if i > 0 {
				mergedContent += "\n\n"
			}
			mergedContent += c
		}
	}

	merged := map[string]interface{}{"role": "system", "content": mergedContent}
	return append([]map[string]interface{}{merged}, otherMsgs...)
}

func (h *Handler) chatToResponses(payload map[string]interface{}, model string) map[string]interface{} {
	result := map[string]interface{}{"model": model}

	if messages, ok := payload["messages"].([]interface{}); ok {
		input := make([]map[string]interface{}, 0, len(messages))
		for _, msg := range messages {
			if m, ok := msg.(map[string]interface{}); ok {
				role := "user"
				if r, ok := m["role"].(string); ok {
					role = normalizeRole(r)
				}
				content := ""
				if c, ok := m["content"].(string); ok {
					content = c
				}
				input = append(input, map[string]interface{}{"role": role, "content": content})
			}
		}
		input = coalesceSystemMessages(input)
		result["input"] = input
	}

	for k, v := range payload {
		if k != "model" && k != "messages" && k != "stream" {
			result[k] = v
		}
	}
	return result
}

func (h *Handler) responsesToChatCompletion(resp map[string]interface{}, model string) map[string]interface{} {
	createdAt := int(time.Now().Unix())
	if c, ok := resp["created_at"].(float64); ok {
		createdAt = int(c)
	}

	result := map[string]interface{}{
		"id":      resp["id"],
		"object":  "chat.completion",
		"created": createdAt,
		"model":   model,
		"choices": []map[string]interface{}{},
	}

	if output, ok := resp["output"].([]interface{}); ok {
		for _, item := range output {
			if msg, ok := item.(map[string]interface{}); ok {
				if msg["type"] == "message" {
					content := ""
					if c, ok := msg["content"].([]interface{}); ok {
						for _, block := range c {
							if b, ok := block.(map[string]interface{}); ok {
								if t, ok := b["text"].(string); ok {
									content += t
								}
							}
						}
					}
					role := "assistant"
					if r, ok := msg["role"].(string); ok {
						role = r
					}
					choice := map[string]interface{}{
						"index": 0,
						"message": map[string]interface{}{
							"role":    role,
							"content": content,
						},
						"finish_reason": "stop",
					}
					result["choices"] = []map[string]interface{}{choice}
					break
				}
			}
		}
	}

	if usage, ok := resp["usage"].(map[string]interface{}); ok {
		result["usage"] = usage
	}

	return result
}

func (h *Handler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	h.sendJSON(w, 200, map[string]string{
		"status":            "ok",
		"upstream_base_v1":  h.config.UpstreamBaseURL,
		"responses_endpoint": h.responsesURL,
	})
}

func (h *Handler) HandleModels(w http.ResponseWriter, r *http.Request) {
	status, body, _ := h.proxyRequest("GET", h.modelsURL, nil)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(body)
}

func (h *Handler) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	bodyBytes, _ := io.ReadAll(r.Body)
	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		h.sendJSON(w, 400, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}

	model := h.config.DefaultModel
	if m, ok := payload["model"].(string); ok && m != "" {
		model = m
	}

	if s, ok := payload["stream"].(bool); ok && s {
		h.sendJSON(w, 400, map[string]string{"error": "stream=true is not supported"})
		return
	}

	converted := h.chatToResponses(payload, model)
	convertedBytes, _ := json.Marshal(converted)

	status, respBody, err := h.proxyRequest("POST", h.responsesURL, convertedBytes)
	if err != nil {
		h.sendJSON(w, 502, map[string]string{"error": fmt.Sprintf("upstream failed: %v", err)})
		return
	}

	var respData map[string]interface{}
	if err := json.Unmarshal(respBody, &respData); err == nil {
		chatResp := h.responsesToChatCompletion(respData, model)
		newBody, _ := json.Marshal(chatResp)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write(newBody)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(respBody)
}

func (h *Handler) HandleResponses(w http.ResponseWriter, r *http.Request) {
	bodyBytes, _ := io.ReadAll(r.Body)
	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		h.sendJSON(w, 400, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}

	model := h.config.DefaultModel
	if m, ok := payload["model"].(string); ok && m != "" {
		model = m
	}
	payload["model"] = model

	payloadBytes, _ := json.Marshal(payload)
	status, respBody, err := h.proxyRequest("POST", h.responsesURL, payloadBytes)
	if err != nil {
		h.sendJSON(w, 502, map[string]string{"error": fmt.Sprintf("upstream failed: %v", err)})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(respBody)
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/")

	switch {
	case path == "health" || path == "health/":
		h.HandleHealth(w, r)
	case path == "v1/models" || path == "models":
		h.HandleModels(w, r)
	case r.Method == "POST" && (path == "v1/chat/completions" || path == "chat/completions"):
		h.HandleChatCompletions(w, r)
	case r.Method == "POST" && (path == "v1/responses" || path == "responses"):
		h.HandleResponses(w, r)
	default:
		h.sendJSON(w, 404, map[string]string{"error": "not_found"})
	}
}

func main() {
	cfg := loadConfig()
	initClient()
	handler := NewHandler(cfg)

	fmt.Printf("[bridge] ready upstream=%s model=%s\n", cfg.UpstreamBaseURL, cfg.DefaultModel)

	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	fmt.Printf("[bridge] listening on %s\n", addr)

	server := &http.Server{Addr: addr, Handler: handler}

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("[bridge] server error: %v\n", err)
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	server.Close()
}