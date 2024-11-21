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

// MaskPosition returns the position of the first [MASK] token in the sequence
func (t *BERTTokenizer) MaskPosition(tokens []string) int {
	for i, token := range tokens {
		if token == t.specialTokens.MASK {
			return i
		}
	}
	return -1
}

// Labels returns the labels for the vocabulary
func (t *BERTTokenizer) Labels() map[int]string {
	return t.labels
}

// VocabSize returns the size of the vocabulary
func (t *BERTTokenizer) VocabSize() int {
	return len(t.vocab)
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
