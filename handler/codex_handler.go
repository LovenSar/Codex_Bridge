package codex_chat

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strconv"
	"time"
)

// ============================================================================
// Handler - HTTP API for Codex Chat Plugin
// ============================================================================

// Handler handles HTTP requests for Codex chat plugin
type Handler struct {
	config *RuntimeConfig
}

// NewHandler creates a new Codex chat handler
func NewHandler(cfg *RuntimeConfig) *Handler {
	return &Handler{config: cfg}
}

// HandleStatus returns plugin status
func (h *Handler) HandleStatus(w http.ResponseWriter, r *http.Request) {
	status := GetStatus()
	
	response := map[string]interface{}{
		"plugin":     "codex_chat",
		"status":     string(status),
		"enabled":    h.config.Enabled,
		"preferred":  h.config.Preferred,
		"bridge_url": fmt.Sprintf("http://%s:%d", h.config.BridgeHost, h.config.BridgePort),
		"upstream":   h.config.UpstreamURL,
		"model":      h.config.Model,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// HandleToggle enables/disables the plugin
func (h *Handler) HandleToggle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request struct {
		Enabled bool `json:"enabled"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	h.config.Enabled = request.Enabled
	
	if request.Enabled {
		if err := Start(); err != nil {
			http.Error(w, fmt.Sprintf("Failed to start: %v", err), http.StatusInternalServerError)
			return
		}
	} else {
		if err := Stop(); err != nil {
			http.Error(w, fmt.Sprintf("Failed to stop: %v", err), http.StatusInternalServerError)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"enabled": strconv.FormatBool(request.Enabled),
	})
}

// HandleUpstreamCheck checks upstream LLM connectivity
func (h *Handler) HandleUpstreamCheck(w http.ResponseWriter, r *http.Request) {
	if err := CheckUpstream(); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"upstream_reachable": false,
			"error":              err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"upstream_reachable": true,
	})
}

// HandleBridgeHealth checks bridge connectivity
func (h *Handler) HandleBridgeHealth(w http.ResponseWriter, r *http.Request) {
	bridgeURL := fmt.Sprintf("http://%s:%d/health", h.config.BridgeHost, h.config.BridgePort)
	
	resp, err := http.Get(bridgeURL)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"bridge_reachable": false,
			"error":            err.Error(),
		})
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"bridge_reachable": true,
		"bridge_url":       bridgeURL,
	})
}

// RegisterRoutes registers HTTP routes for this plugin
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/plugins/codex_chat/status", h.HandleStatus)
	mux.HandleFunc("/plugins/codex_chat/toggle", h.HandleToggle)
	mux.HandleFunc("/plugins/codex_chat/upstream", h.HandleUpstreamCheck)
	mux.HandleFunc("/plugins/codex_chat/health", h.HandleBridgeHealth)
}

// ============================================================================
// Plugin Manager Integration
// ============================================================================

var globalHandler *Handler

// RegisterWithMux registers the plugin with the given serve mux
func RegisterWithMux(mux *http.ServeMux) error {
	if err := Init(); err != nil {
		return fmt.Errorf("failed to init plugin: %w", err)
	}
	
	if err := Start(); err != nil {
		// Don't fail if bridge is already running externally
		fmt.Printf("[codex_chat] Warning: %v\n", err)
	}
	
	globalHandler = NewHandler(GetConfig())
	globalHandler.RegisterRoutes(mux)
	
	return nil
}

// UnregisterFromMux removes routes from the serve mux
func UnregisterFromMux(mux *http.ServeMux) error {
	Stop()
	globalHandler = nil
	return nil
}

// GetHandler returns the global handler instance
func GetHandler() *Handler {
	return globalHandler
}