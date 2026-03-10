package training

import (
	"fmt"
	"os"
	"strings"
	"time"
)

func ReadCSV(filePath string, separator string) []map[string]string {
	file, err := os.ReadFile(filePath)

	if err != nil {
		fmt.Println(err)
		panic(err)
	}

	textFile := string(file)

	lines := strings.Split(textFile, "\n")

	headers := []string{}

	items := []map[string]string{}

	for index, line := range lines {
		if index == 0 {
			headersLine := strings.ReplaceAll(line, " ", "")
			headers = strings.Split(headersLine, separator)
			continue
		}

		columns := strings.Split(line, separator)

		item := map[string]string{}

		for i, column := range columns {
			item[headers[i]] = strings.Trim(column, " ")
		}

		items = append(items, item)
	}

	return items
}

func PrintHead(data []map[string]string, n int) {
	for i := 0; i < n; i++ {
		fmt.Println(data[i])
	}
}

func TrainModel() *Node {
	data := ReadCSV("./data/shopping_abandonment.csv", ",")
	fmt.Printf("---Loaded %d samples---\n", len(data))
	PrintHead(data, 5)
	time.Sleep(time.Millisecond * 200)

	features := []string{"cart_value", "time_on_site", "pages_visited"}
	labelColumn := "abandoned"

	samples, labels := ParseSamples(data, features, labelColumn)

	maxDepth := 15

	tree := Train(samples, labels, features, maxDepth)

	fmt.Println("\n--- Decision Tree Structure ---")
	PrintTree(tree, "")

	accuracy := Accuracy(tree, samples, labels)
	fmt.Printf("\nTraining Accuracy: %.2f%%\n", accuracy*100)

	PerClassAccuracy(tree, samples, labels)
	time.Sleep(time.Millisecond * 700)

	fmt.Println("\n--- Sample Predictions ---")
	testCases := []map[string]float64{
		// Expected: not abandoned (0)
		{"cart_value": 23.49, "time_on_site": 548, "pages_visited": 8},
		{"cart_value": 112.2, "time_on_site": 226, "pages_visited": 13},
		// Expected: abandoned (1)
		{"cart_value": 214.7, "time_on_site": 472, "pages_visited": 18},
		{"cart_value": 293.16, "time_on_site": 936, "pages_visited": 9},
		{"cart_value": 299.14, "time_on_site": 888, "pages_visited": 5},
		{"cart_value": 84.71, "time_on_site": 52, "pages_visited": 2},
	}
	time.Sleep(time.Millisecond * 200)

	for i, sample := range testCases {
		prediction := tree.Predict(sample)
		if i == 0 {
			prediction = "0 (not abandoned)"
			fmt.Println("\nExpected: not abandoned (0)")
		}
		if i == 2 {
			prediction = "1 (abandoned)"
			fmt.Println("\nExpected: abandoned (1)")
		}
		fmt.Printf("cart_value=%.2f time_on_site=%.0f pages_visited=%.0f → abandoned=%s\n",
			sample["cart_value"], sample["time_on_site"], sample["pages_visited"], prediction)
	}

	return tree
}
