package tokenizer

import (
	"bufio"
	"embed"
	"fmt"
	"regexp"
	"strings"
)

//go:embed vocabs/bert.txt
var bertVocabFS embed.FS

type BERTTokenizer struct {
	vocab         map[string]int
	labels        map[int]string
	specialTokens SpecialTokens
}

func NewBERTTokenizer() (*BERTTokenizer, error) {
	vocab, err := loadVocabFromEmbed()
	if err != nil {
		return nil, err
	}

	labels := make(map[int]string, len(vocab))
	for token, id := range vocab {
		labels[id] = token
	}

	return &BERTTokenizer{
		vocab:         vocab,
		labels:        labels,
		specialTokens: DefaultSpecialTokens(),
	}, nil
}

func (t *BERTTokenizer) Encode(text string, maxLength int) (*TokenizerOutput, error) {
	pattern := regexp.MustCompile(`\[[^\[\]]+\]|\w+|[^\w\s]+`)
	tokens := pattern.FindAllString(text, -1)
	wordpieceTokens := []string{}

	for _, token := range tokens {
		if strings.HasPrefix(token, "[") && strings.HasSuffix(token, "]") {
			// Handle special tokens case-insensitively
			if canonical, isSpecial := t.specialTokens.IsSpecialToken(token); isSpecial {
				wordpieceTokens = append(wordpieceTokens, canonical)
				continue
			}
		}

		token = strings.ToLower(token)
		if _, ok := t.vocab[token]; ok {
			wordpieceTokens = append(wordpieceTokens, token)
		} else {
			subTokens := WordPiece(t.vocab, t.specialTokens, token)
			wordpieceTokens = append(wordpieceTokens, subTokens...)
		}
	}

	inputIds := []int64{int64(t.vocab[t.specialTokens.CLS])}
	finalTokens := []string{t.specialTokens.CLS}

	for _, token := range wordpieceTokens {
		if id, ok := t.vocab[token]; ok {
			inputIds = append(inputIds, int64(id))
			finalTokens = append(finalTokens, token)
		} else {
			inputIds = append(inputIds, int64(t.vocab[t.specialTokens.UNK]))
			finalTokens = append(finalTokens, t.specialTokens.UNK)
		}
	}

	inputIds = append(inputIds, int64(t.vocab[t.specialTokens.SEP]))
	finalTokens = append(finalTokens, t.specialTokens.SEP)

	attentionMask := make([]int64, len(inputIds))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	if len(inputIds) > maxLength {
		inputIds = inputIds[:maxLength]
		finalTokens = finalTokens[:maxLength]
		attentionMask = attentionMask[:maxLength]
	} else {
		for len(inputIds) < maxLength {
			inputIds = append(inputIds, int64(t.vocab[t.specialTokens.PAD]))
			attentionMask = append(attentionMask, 0)
			finalTokens = append(finalTokens, t.specialTokens.PAD)
		}
	}

	return &TokenizerOutput{
		InputIds:      inputIds,
		AttentionMask: attentionMask,
		Tokens:        finalTokens,
	}, nil
}

// MaskLogits represents the logits for masked tokens
type MaskLogits struct {
	Position int       // Position of the mask token
	Logits   []float32 // Logits for the mask token
}

// MaskLogits extracts logits for all mask tokens in the sequence
func (t *BERTTokenizer) MaskLogits(tokens []string, logits []float32) ([]MaskLogits, error) {
	if len(logits)%len(tokens) != 0 {
		return nil, fmt.Errorf("logits length (%d) is not a multiple of tokens length (%d)", len(logits), len(tokens))
	}

	vocabSize := len(t.vocab)

	var maskLogits []MaskLogits
	for pos, token := range tokens {
		if token == t.specialTokens.MASK {
			start := pos * vocabSize
			end := start + vocabSize
			if end > len(logits) {
				return nil, fmt.Errorf("logits array too short for mask at position %d", pos)
			}

			maskLogits = append(maskLogits, MaskLogits{
				Position: pos,
				Logits:   logits[start:end],
			})
		}
	}

	return maskLogits, nil
}

// Labels returns the labels for the vocabulary
func (t *BERTTokenizer) Labels() map[int]string {
	return t.labels
}

func loadVocabFromEmbed() (map[string]int, error) {
	vocab := make(map[string]int)

	file, err := bertVocabFS.Open("vocabs/bert.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to open embedded vocab file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	id := 0
	for scanner.Scan() {
		token := strings.TrimSpace(scanner.Text())
		if token != "" {
			vocab[token] = id
			id++
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading vocab file: %w", err)
	}

	return vocab, nil
}
