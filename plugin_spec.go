package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ============================================================================
// Plugin Metadata - Implements plugin_spec.go interfaces
// ============================================================================

// PluginInfo implements the standard plugin interface
var PluginInfo = PluginMetadata{
	Name:        "codex_chat",
	DisplayName: "Code Bridge",
	Version:     "0.1.0-alpha",
	Author:      "Code_Bridge",
	Description: "OpenAI-compatible HTTP bridge for Codex CLI and upstream LLM services",
	Category:    "chat_backend",
	Languages:   []PluginLanguage{LangGo, LangPython},
	Status:      PluginStatusLoading,
	SkillPath:   "bridge",
	Options: []PluginOption{
		{
			Name:        "enabled",
			DisplayName: "Enable Codex Chat",
			Type:        "bool",
			Default:     true,
			Required:    false,
			Description: "Enable Codex as chat backend",
		},
		{
			Name:        "preferred",
			DisplayName: "Preferred Backend",
			Type:        "bool",
			Default:     false,
			Required:    false,
			Description: "Prefer this bridge as the primary chat backend when multiple backends are available",
		},
		{
			Name:        "bridge_port",
			DisplayName: "Bridge Port",
			Type:        "int",
			Default:     18081,
			Required:    false,
			Description: "Codex bridge listening port",
		},
		{
			Name:        "bridge_host",
			DisplayName: "Bridge Host",
			Type:        "string",
			Default:     "127.0.0.1",
			Required:    false,
			Description: "Codex bridge listening host",
		},
		{
			Name:        "upstream_url",
			DisplayName: "Upstream URL",
			Type:        "string",
			Default:     "http://100.122.242.51:8000/v1",
			Required:    false,
			Description: "Upstream LLM API endpoint",
		},
		{
			Name:        "model",
			DisplayName: "Model Name",
			Type:        "string",
			Default:     "Qwen/Qwen3.5-27B",
			Required:    false,
			Description: "Default LLM model name",
		},
	},
}

// PluginMetadata holds plugin metadata
type PluginMetadata struct {
	Name        string           `json:"name"`
	DisplayName string           `json:"display_name"`
	Version     string           `json:"version"`
	Author      string           `json:"author"`
	Description string           `json:"description"`
	Category    string           `json:"category"`
	Languages   []PluginLanguage `json:"languages"`
	Status      PluginStatus     `json:"status"`
	Options     []PluginOption    `json:"options"`
	SkillPath   string           `json:"skill_path"`
	LoadedAt    time.Time        `json:"loaded_at"`
}

// RuntimeConfig holds runtime configuration
type RuntimeConfig struct {
	Enabled         bool          `json:"enabled"`
	Preferred       bool          `json:"preferred"`
	BridgeHost      string        `json:"bridge_host"`
	BridgePort      int           `json:"bridge_port"`
	UpstreamURL     string        `json:"upstream_url"`
	Model           string        `json:"model"`
	UpstreamAPIKey  string        `json:"upstream_api_key"`
	BridgePID       int           `json:"bridge_pid"`
	BridgeProcess   *os.Process
}

// ============================================================================
// Plugin Lifecycle Functions
// ============================================================================

var runtimeCfg *RuntimeConfig

// Init initializes the plugin
func Init() error {
	runtimeCfg = &RuntimeConfig{
		Enabled:        getEnvBool("CODEX_CHAT_ENABLED", true),
		Preferred:      getEnvBool("CODEX_CHAT_PREFERRED", false),
		BridgeHost:     getEnv("CODEX_CHAT_BRIDGE_HOST", "127.0.0.1"),
		BridgePort:     getEnvInt("CODEX_CHAT_BRIDGE_PORT", 18081),
		UpstreamURL:    getEnv("CODEX_UPSTREAM_BASE_URL", "http://100.122.242.51:8000/v1"),
		Model:          getEnv("CODEX_MODEL", "Qwen/Qwen3.5-27B"),
		UpstreamAPIKey: getEnv("CODEX_UPSTREAM_API_KEY", "dummy"),
	}

	PluginInfo.Options = updateOptionsFromConfig(PluginInfo.Options, runtimeCfg)
	return nil
}

// Start starts the plugin services
func Start() error {
	if !runtimeCfg.Enabled {
		PluginInfo.Status = PluginStatusDisabled
		return nil
	}

	// Check if bridge is already running
	if isBridgeRunning() {
		PluginInfo.Status = PluginStatusReady
		return nil
	}

	// Start bridge
	if err := startBridge(); err != nil {
		PluginInfo.Status = PluginStatusError
		return fmt.Errorf("failed to start bridge: %w", err)
	}

	PluginInfo.Status = PluginStatusReady
	PluginInfo.LoadedAt = time.Now()
	return nil
}

// Stop stops the plugin services
func Stop() error {
	if runtimeCfg.BridgeProcess != nil {
		runtimeCfg.BridgeProcess.Kill()
		runtimeCfg.BridgeProcess = nil
	}
	PluginInfo.Status = PluginStatusDisabled
	return nil
}

// GetStatus returns current plugin status
func GetStatus() PluginStatus {
	if !runtimeCfg.Enabled {
		return PluginStatusDisabled
	}
	if isBridgeRunning() {
		return PluginStatusReady
	}
	return PluginStatusError
}

// GetConfig returns current runtime configuration
func GetConfig() *RuntimeConfig {
	return runtimeCfg
}

// ============================================================================
// Bridge Management
// ============================================================================

func getBridgeScriptPath() string {
	return "bridge/chat_bridge.py"
}

func isBridgeRunning() bool {
	// Check if port is listening
	cmd := exec.Command("powershell", "-NoProfile", "-Command",
		fmt.Sprintf(`netstat -ano | findstr ":%d.*LISTENING"`, runtimeCfg.BridgePort))
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	return len(output) > 0
}

func startBridge() error {
	scriptPath := getBridgeScriptPath()
	
	// Check if script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		scriptPath = "chat_bridge.py"
		if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
			return fmt.Errorf("bridge script not found (expected bridge/chat_bridge.py or chat_bridge.py)")
		}
	}

	cmd := exec.Command("python", scriptPath,
		"--host", runtimeCfg.BridgeHost,
		"--port", fmt.Sprintf("%d", runtimeCfg.BridgePort),
		"--upstream-base-url", runtimeCfg.UpstreamURL,
		"--upstream-api-key", runtimeCfg.UpstreamAPIKey,
		"--default-model", runtimeCfg.Model,
	)

	// Start in background
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start bridge: %w", err)
	}

	runtimeCfg.BridgeProcess = cmd.Process
	
	// Wait for bridge to be ready
	time.Sleep(2 * time.Second)
	
	if !isBridgeRunning() {
		return fmt.Errorf("bridge did not start successfully")
	}

	return nil
}

// CheckUpstream checks if upstream LLM is reachable
func CheckUpstream() error {
	resp, err := http.Get(runtimeCfg.UpstreamURL + "/models")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return fmt.Errorf("upstream returned status %d", resp.StatusCode)
	}
	return nil
}

// ============================================================================
// Utility Functions
// ============================================================================

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return strings.ToLower(value) == "true" || value == "1" || value == "yes"
}

func getEnvInt(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	intValue, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return intValue
}

func updateOptionsFromConfig(options []PluginOption, cfg *RuntimeConfig) []PluginOption {
	for i := range options {
		switch options[i].Name {
		case "enabled":
			options[i].Default = cfg.Enabled
		case "preferred":
			options[i].Default = cfg.Preferred
		case "bridge_port":
			options[i].Default = cfg.BridgePort
		case "bridge_host":
			options[i].Default = cfg.BridgeHost
		case "upstream_url":
			options[i].Default = cfg.UpstreamURL
		case "model":
			options[i].Default = cfg.Model
		}
	}
	return options
}

// ============================================================================
// JSON Serialization for API
// ============================================================================

func (p *PluginMetadata) ToJSON() string {
	data, _ := json.MarshalIndent(p, "", "  ")
	return string(data)
}

func (c *RuntimeConfig) ToJSON() string {
	data, _ := json.MarshalIndent(c, "", "  ")
	return string(data)
}