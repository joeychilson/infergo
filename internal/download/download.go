package download

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
)

func DownloadFile(ctx context.Context, url string, destPath string) (string, error) {
	client := http.DefaultClient

	tmpFile := destPath + ".download"
	defer os.Remove(tmpFile)

	f, err := os.Create(tmpFile)
	if err != nil {
		return "", fmt.Errorf("failed to create temporary file: %w", err)
	}
	defer f.Close()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	if _, err := io.Copy(f, resp.Body); err != nil {
		return "", fmt.Errorf("failed to save file: %w", err)
	}

	if err := os.Rename(tmpFile, destPath); err != nil {
		return "", fmt.Errorf("failed to move downloaded file: %w", err)
	}
	return destPath, nil
}
