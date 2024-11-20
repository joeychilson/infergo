package onnx

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/joeychilson/infergo/internal/archive"
	"github.com/joeychilson/infergo/internal/download"
	ort "github.com/yalue/onnxruntime_go"
)

const currentVersion = "1.20.0"

// Runtime manages ONNX Runtime initialization and configuration
type Runtime struct {
	gpu         bool
	cachePath   string
	libraryPath string
}

// Option is a functional option for configuring Runtime
type Option func(*Runtime)

// WithGPU enables GPU support
func WithGPU(enabled bool) Option {
	return func(r *Runtime) {
		r.gpu = enabled
	}
}

// WithCachePath sets the cache directory
func WithCachePath(path string) Option {
	return func(r *Runtime) {
		r.cachePath = path
	}
}

// WithLibraryPath sets a direct path to the ONNX Runtime library
func WithLibraryPath(path string) Option {
	return func(r *Runtime) {
		r.libraryPath = path
	}
}

// New creates a new ONNX Runtime manager
func New(ctx context.Context, opts ...Option) (*Runtime, error) {
	runtime := &Runtime{
		cachePath: filepath.Join(os.TempDir(), "goml"),
		gpu:       false,
	}

	for _, opt := range opts {
		opt(runtime)
	}

	libPath, err := runtime.EnsureRuntime(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to ensure runtime: %w", err)
	}

	ort.SetSharedLibraryPath(libPath)

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize environment: %w", err)
	}
	return runtime, nil
}

// RuntimeInfo contains ONNX Runtime specific information
type RuntimeInfo struct {
	Version     string
	OS          string
	Arch        string
	GPU         bool
	LibraryName string
}

// GetRuntimeInfo returns information about the current runtime
func (r *Runtime) RuntimeInfo() *RuntimeInfo {
	info := &RuntimeInfo{Version: currentVersion, GPU: r.gpu}

	switch runtime.GOOS {
	case "windows":
		info.OS = "win"
		info.LibraryName = "onnxruntime.dll"
	case "darwin":
		info.OS = "osx"
		info.LibraryName = fmt.Sprintf("libonnxruntime.%s.dylib", info.Version)
	default:
		info.OS = "linux"
		info.LibraryName = fmt.Sprintf("libonnxruntime.so.%s", info.Version)
	}

	switch runtime.GOARCH {
	case "amd64":
		if info.OS == "linux" {
			info.Arch = "x64"
		} else if info.OS == "osx" {
			info.Arch = "x86_64"
		} else {
			info.Arch = "x64"
		}
	case "arm64":
		if info.OS == "linux" {
			info.Arch = "aarch64"
		} else {
			info.Arch = "arm64"
		}
	case "386":
		if info.OS == "win" {
			info.Arch = "x86"
		}
	}
	return info
}

// RuntimeURL returns the download URL for a specific runtime
func (r *Runtime) RuntimeURL(info *RuntimeInfo) string {
	base := fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/", info.Version)

	name := fmt.Sprintf("onnxruntime-%s-%s", info.OS, info.Arch)

	if info.GPU && (info.OS == "linux" || info.OS == "win") && info.Arch == "x64" {
		name += "-gpu"
	}

	name += fmt.Sprintf("-%s", info.Version)
	if info.OS == "win" {
		name += ".zip"
	} else {
		name += ".tgz"
	}
	return base + name
}

// EnsureRuntime downloads and extracts the ONNX Runtime library
func (r *Runtime) EnsureRuntime(ctx context.Context) (string, error) {
	runtime := r.RuntimeInfo()

	if r.libraryPath != "" {
		if filepath.Ext(r.libraryPath) != filepath.Ext(runtime.LibraryName) {
			return "", fmt.Errorf("specified library invalid for current platform")
		}
		if _, err := os.Stat(r.libraryPath); err != nil {
			return "", fmt.Errorf("specified library path does not exist: %w", err)
		}
		return r.libraryPath, nil
	}

	libDir := filepath.Join(r.cachePath, "runtime")
	if err := os.MkdirAll(libDir, 0755); err != nil {
		return "", err
	}

	libPath := filepath.Join(libDir, runtime.LibraryName)
	if _, err := os.Stat(libPath); err == nil {
		return libPath, nil
	}

	url := r.RuntimeURL(runtime)

	targetPath := filepath.Join(r.cachePath, "runtime", filepath.Base(url))
	if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
		return "", err
	}

	if _, err := os.Stat(targetPath); err != nil {
		targetPath, err = download.DownloadFile(ctx, url, targetPath)
		if err != nil {
			return "", fmt.Errorf("failed to download runtime: %w", err)
		}
	}

	if strings.HasSuffix(targetPath, ".zip") {
		if err := archive.ExtractFromZip(targetPath, libPath, runtime.LibraryName); err != nil {
			return "", fmt.Errorf("failed to extract runtime: %w", err)
		}
	} else {
		if err := archive.ExtractFromTarGz(targetPath, libPath, runtime.LibraryName); err != nil {
			return "", fmt.Errorf("failed to extract runtime: %w", err)
		}
	}

	if err := os.Remove(targetPath); err != nil {
		return "", fmt.Errorf("failed to remove archive: %w", err)
	}
	return libPath, nil
}

// Version returns the current ONNX Runtime version
func (r *Runtime) Version() string {
	return ort.GetVersion()
}

// Close cleans up ONNX Runtime resources
func (r *Runtime) Close() error {
	return ort.DestroyEnvironment()
}
