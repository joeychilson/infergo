package archive

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"strings"
)

// ExtractFromZip extracts a specific file from a zip archive
func ExtractFromZip(archivePath, destPath, targetFile string) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer reader.Close()

	for _, file := range reader.File {
		if strings.HasSuffix(file.Name, targetFile) {
			reader, err := file.Open()
			if err != nil {
				return err
			}
			defer reader.Close()

			writer, err := os.Create(destPath)
			if err != nil {
				return err
			}
			defer writer.Close()

			_, err = io.Copy(writer, reader)
			return err
		}
	}
	return fmt.Errorf("file %s not found in archive", targetFile)
}

// ExtractFromTarGz extracts a specific file from a tar.gz archive
func ExtractFromTarGz(archivePath, destPath, targetFile string) error {
	file, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if strings.HasSuffix(header.Name, targetFile) {
			writer, err := os.Create(destPath)
			if err != nil {
				return err
			}
			defer writer.Close()

			_, err = io.Copy(writer, tr)
			return err
		}
	}
	return fmt.Errorf("file %s not found in archive", targetFile)
}
