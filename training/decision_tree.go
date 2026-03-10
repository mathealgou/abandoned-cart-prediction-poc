package training

import (
	"fmt"
	"math"
	"strconv"
)

type Node struct {
	IsLeaf     bool
	Prediction string

	Feature   string
	Threshold float64
	Left      *Node
	Right     *Node
}

func (n *Node) Predict(sample map[string]float64) string {
	if n.IsLeaf {
		return n.Prediction
	}

	if sample[n.Feature] <= n.Threshold {
		return n.Left.Predict(sample)
	}

	return n.Right.Predict(sample)
}

func giniImpurity(labels []string) float64 {
	if len(labels) == 0 {
		return 0.0
	}

	counts := map[string]int{}
	for _, l := range labels {
		counts[l]++
	}

	impurity := 1.0
	total := float64(len(labels))
	for _, count := range counts {
		prob := float64(count) / total
		impurity -= prob * prob
	}

	return impurity
}

func weightedGini(left, right []string) float64 {
	total := float64(len(left) + len(right))
	leftWeight := float64(len(left)) / total
	rightWeight := float64(len(right)) / total

	return leftWeight*giniImpurity(left) + rightWeight*giniImpurity(right)
}

func majorityLabel(labels []string) string {
	counts := map[string]int{}
	for _, l := range labels {
		counts[l]++
	}

	best := ""
	bestCount := 0
	for label, count := range counts {
		if count > bestCount {
			bestCount = count
			best = label
		}
	}

	return best
}

// bestSplit finds the best feature and threshold to split the data to minimize the Gini impurity.
func bestSplit(samples []map[string]float64, labels []string, features []string) (string, float64) {
	bestFeature := ""
	bestThreshold := 0.0
	bestGini := math.Inf(1)

	for _, feature := range features {
		values := map[float64]bool{}
		for _, sample := range samples {
			values[sample[feature]] = true
		}

		for threshold := range values {
			leftLabels := []string{}
			rightLabels := []string{}

			for i, sample := range samples {
				if sample[feature] <= threshold {
					leftLabels = append(leftLabels, labels[i])
				} else {
					rightLabels = append(rightLabels, labels[i])
				}
			}

			if len(leftLabels) == 0 || len(rightLabels) == 0 {
				continue
			}

			gini := weightedGini(leftLabels, rightLabels)
			if gini < bestGini {
				bestGini = gini
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold
}

// Recursively generates a decision tree by splitting the data based on the best feature and threshold until a stopping condition is met (e.g., max depth or pure node).
func Train(samples []map[string]float64, labels []string, features []string, maxDepth int) *Node {
	if len(labels) == 0 {
		return &Node{IsLeaf: true, Prediction: "unknown"}
	}

	if maxDepth == 0 || giniImpurity(labels) == 0 {
		return &Node{IsLeaf: true, Prediction: majorityLabel(labels)}
	}

	feature, threshold := bestSplit(samples, labels, features)

	if feature == "" {
		return &Node{IsLeaf: true, Prediction: majorityLabel(labels)}
	}

	leftSamples, leftLabels := []map[string]float64{}, []string{}
	rightSamples, rightLabels := []map[string]float64{}, []string{}

	for i, sample := range samples {
		if sample[feature] <= threshold {
			leftSamples = append(leftSamples, sample)
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightSamples = append(rightSamples, sample)
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return &Node{
		Feature:   feature,
		Threshold: threshold,
		Left:      Train(leftSamples, leftLabels, features, maxDepth-1),
		Right:     Train(rightSamples, rightLabels, features, maxDepth-1),
	}
}

func ParseSamples(data []map[string]string, features []string, labelColumn string) ([]map[string]float64, []string) {
	samples := []map[string]float64{}
	labels := []string{}

	for _, row := range data {
		sample := map[string]float64{}
		valid := true

		for _, feature := range features {
			val, err := strconv.ParseFloat(row[feature], 64)
			if err != nil {
				valid = false
				break
			}
			sample[feature] = val
		}

		if valid {
			samples = append(samples, sample)
			labels = append(labels, row[labelColumn])
		}
	}

	return samples, labels
}

func Accuracy(tree *Node, samples []map[string]float64, labels []string) float64 {
	correct := 0
	for i, sample := range samples {
		if tree.Predict(sample) == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(labels))
}

func PerClassAccuracy(tree *Node, samples []map[string]float64, labels []string) {
	classTotals := map[string]int{}
	classCorrect := map[string]int{}

	for i, sample := range samples {
		actual := labels[i]
		classTotals[actual]++
		if tree.Predict(sample) == actual {
			classCorrect[actual]++
		}
	}

	fmt.Println("Per-class accuracy:")
	for label, total := range classTotals {
		acc := float64(classCorrect[label]) / float64(total) * 100
		fmt.Printf("  class %s: %d/%d = %.2f%%\n", label, classCorrect[label], total, acc)
	}
}

var leafCount int = 0

func PrintTree(n *Node, indent string) {
	if n.IsLeaf {
		// fmt.Printf("%s→ predict: %s\n", indent, n.Prediction)
		// clear the line and print the leaf count
		fmt.Printf("\rleafCount: %d", leafCount)
		leafCount++
		return
	}
	PrintTree(n.Left, indent+"  ")
	PrintTree(n.Right, indent+"  ")
}
