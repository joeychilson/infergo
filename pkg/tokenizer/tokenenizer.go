package tokenizer

import "strings"

// SpecialTokens represents the special tokens used by the tokenizer
type SpecialTokens struct {
	PAD  string
	UNK  string
	CLS  string
	SEP  string
	MASK string
}

// DefaultSpecialTokens returns the default special tokens for BERT
func DefaultSpecialTokens() SpecialTokens {
	return SpecialTokens{
		PAD:  "[PAD]",
		UNK:  "[UNK]",
		CLS:  "[CLS]",
		SEP:  "[SEP]",
		MASK: "[MASK]",
	}
}

// SpecialTokenMap returns a map of case-insensitive special tokens to their canonical form
func (st SpecialTokens) SpecialTokenMap() map[string]string {
	return map[string]string{
		strings.ToUpper(st.PAD):  st.PAD,
		strings.ToUpper(st.UNK):  st.UNK,
		strings.ToUpper(st.CLS):  st.CLS,
		strings.ToUpper(st.SEP):  st.SEP,
		strings.ToUpper(st.MASK): st.MASK,
	}
}

// IsSpecialToken checks if a token is a special token (case-insensitive)
func (st SpecialTokens) IsSpecialToken(token string) (string, bool) {
	upperToken := strings.ToUpper(token)
	if canonical, ok := st.SpecialTokenMap()[upperToken]; ok {
		return canonical, true
	}
	return "", false
}

// TokenizerOutput represents the output of the tokenizer
type TokenizerOutput struct {
	InputIds      []int64
	AttentionMask []int64
	Tokens        []string
}

func WordPiece(vocab map[string]int, specialTokens SpecialTokens, word string) []string {
	if _, ok := vocab[word]; ok {
		return []string{word}
	}

	tokens := []string{}
	start := 0
	wordLen := len(word)
	for start < wordLen {
		var subword string

		end := wordLen
		found := false

		for end > start {
			substr := word[start:end]
			if start > 0 {
				substr = "##" + substr
			}

			if _, ok := vocab[substr]; ok {
				subword = substr
				found = true
				break
			}
			end--
		}

		if !found {
			return []string{specialTokens.UNK}
		}

		tokens = append(tokens, subword)
		start = end
	}
	return tokens
}
