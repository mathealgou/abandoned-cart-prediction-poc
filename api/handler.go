package api

import (
	"encoding/json"
	"main/training"
	"net/http"
)

type PredictRequest struct {
	CartValue    float64 `json:"cart_value"`
	TimeOnSite   float64 `json:"time_on_site"`
	PagesVisited float64 `json:"pages_visited"`
}

type PredictResponse struct {
	Abandoned    string  `json:"abandoned"`
	CartValue    float64 `json:"cart_value"`
	TimeOnSite   float64 `json:"time_on_site"`
	PagesVisited float64 `json:"pages_visited"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func NewPredictHandler(model *training.Node) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusMethodNotAllowed)
			json.NewEncoder(w).Encode(ErrorResponse{Error: "method not allowed, use POST"})
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(ErrorResponse{Error: "invalid JSON body: " + err.Error()})
			return
		}

		sample := map[string]float64{
			"cart_value":    req.CartValue,
			"time_on_site":  req.TimeOnSite,
			"pages_visited": req.PagesVisited,
		}

		prediction := model.Predict(sample)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(PredictResponse{
			Abandoned:    prediction,
			CartValue:    req.CartValue,
			TimeOnSite:   req.TimeOnSite,
			PagesVisited: req.PagesVisited,
		})
	}
}
