package main

import (
	"fmt"
	"io"
	"log"
	"main/api"
	"main/training"
	"net/http"
	"strings"
	"time"
)

func main() {
	startTime := time.Now()
	model := training.TrainModel()
	fmt.Printf("Model trained in %v\n", time.Since(startTime))

	http.HandleFunc("/predict", api.NewPredictHandler(model))

	addr := ":8080"
	log.Printf("Starting server on %s", addr)
	go func() {
		log.Fatal(http.ListenAndServe(addr, nil))
	}()

	listeningStartTime := time.Now()
	numberOfRequests := 100
	url := fmt.Sprintf("http://localhost%s/predict", addr)
	for i := 0; i < numberOfRequests; i++ {
		sample := map[string]interface{}{
			"cart_value":    i * 10.0,
			"time_on_site":  i * 5.0,
			"pages_visited": i,
		}

		content := fmt.Sprintf(`{"cart_value": %d, "time_on_site": %d, "pages_visited": %d}`, sample["cart_value"], sample["time_on_site"], sample["pages_visited"])

		body := io.NopCloser(strings.NewReader(content))
		res, _ := http.Post(url, "application/json", body)

		body.Close()

		fmt.Printf("Request %d: Status Code: %d\n", i+1, res.StatusCode)
	}

	fmt.Printf("Total time for %d requests: %v\n", numberOfRequests, time.Since(listeningStartTime))
}
