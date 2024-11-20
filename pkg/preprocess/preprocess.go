package preprocess

import (
	"errors"
	"image"
	"math"

	"golang.org/x/image/draw"
)

// ImageData represents preprocessed image data ready for model inference
type ImageData struct {
	Pixels   []float32
	Width    int
	Height   int
	Channels int
	OrigSize image.Point
}

// ResizeMode defines how to handle image resizing
type ResizeMode int

const (
	// ResizeFixed resizes to exact dimensions
	ResizeFixed ResizeMode = iota
	// ResizeAspectFill maintains aspect ratio, fills target size
	ResizeAspectFill
	// ResizeAspectFit maintains aspect ratio, fits within target size
	ResizeAspectFit
	// ResizeWithEdges maintains aspect ratio with min/max edge constraints
	ResizeWithEdges
)

// ProcessImageOptions contains all preprocessing configuration
type ProcessImageOptions struct {
	Width      int
	Height     int
	MinEdge    int
	MaxEdge    int
	Mode       ResizeMode
	Mean       [3]float32
	StdDev     [3]float32
	CenterCrop bool
}

// ProcessImage preprocesses an image according to the specified options
func ProcessImage(img image.Image, opts ProcessImageOptions) (*ImageData, error) {
	if img == nil {
		return nil, errors.New("nil image")
	}

	origSize := image.Point{
		X: img.Bounds().Dx(),
		Y: img.Bounds().Dy(),
	}

	if origSize.X < 1 || origSize.Y < 1 {
		return nil, errors.New("invalid image dimensions")
	}

	width, height := calculateDimensions(origSize.X, origSize.Y, opts)

	resized := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	var processed *image.RGBA
	if opts.CenterCrop {
		processed = centerCrop(resized, opts.Width, opts.Height)
	} else {
		processed = resized
	}

	data := imageToFloat32(processed, opts)

	return &ImageData{
		Pixels:   data,
		Width:    processed.Bounds().Dx(),
		Height:   processed.Bounds().Dy(),
		Channels: 3,
		OrigSize: origSize,
	}, nil
}

func calculateDimensions(origWidth, origHeight int, opts ProcessImageOptions) (newWidth, newHeight int) {
	switch opts.Mode {
	case ResizeFixed:
		return opts.Width, opts.Height

	case ResizeAspectFit:
		scale := math.Min(
			float64(opts.Width)/float64(origWidth),
			float64(opts.Height)/float64(origHeight),
		)
		return int(math.Round(float64(origWidth) * scale)), int(math.Round(float64(origHeight) * scale))

	case ResizeAspectFill:
		scale := math.Max(
			float64(opts.Width)/float64(origWidth),
			float64(opts.Height)/float64(origHeight),
		)
		return int(math.Round(float64(origWidth) * scale)), int(math.Round(float64(origHeight) * scale))

	case ResizeWithEdges:
		if origWidth > origHeight {
			scale := float64(opts.MaxEdge) / float64(origWidth)
			newWidth = opts.MaxEdge
			newHeight = int(math.Round(float64(origHeight) * scale))
			if newHeight > opts.MinEdge {
				scale = float64(opts.MinEdge) / float64(origHeight)
				newHeight = opts.MinEdge
				newWidth = int(math.Round(float64(origWidth) * scale))
			}
		} else {
			scale := float64(opts.MaxEdge) / float64(origHeight)
			newHeight = opts.MaxEdge
			newWidth = int(math.Round(float64(origWidth) * scale))
			if newWidth > opts.MinEdge {
				scale = float64(opts.MinEdge) / float64(origWidth)
				newWidth = opts.MinEdge
				newHeight = int(math.Round(float64(origHeight) * scale))
			}
		}
		return newWidth, newHeight
	}
	return origWidth, origHeight
}

func centerCrop(img *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	bounds := img.Bounds()
	startX := (bounds.Dx() - targetWidth) / 2
	startY := (bounds.Dy() - targetHeight) / 2

	cropped := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	draw.Draw(cropped, cropped.Bounds(), img, image.Point{X: startX, Y: startY}, draw.Src)
	return cropped
}

func imageToFloat32(img *image.RGBA, opts ProcessImageOptions) []float32 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	pixels := make([]float32, 3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			rf := normalizeChannel(r>>8, 0, opts)
			gf := normalizeChannel(g>>8, 1, opts)
			bf := normalizeChannel(b>>8, 2, opts)

			pixels[0*height*width+y*width+x] = rf
			pixels[1*height*width+y*width+x] = gf
			pixels[2*height*width+y*width+x] = bf
		}
	}
	return pixels
}

func normalizeChannel(value uint32, channel int, opts ProcessImageOptions) float32 {
	normalized := float32(value) / 255.0
	normalized = (normalized - opts.Mean[channel]) / opts.StdDev[channel]
	return normalized
}
