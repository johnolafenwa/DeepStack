package response

type ErrorResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
}

type ErrorResponseInternal struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
	Code    int    `json:"code"`
}

type RecognitionResponse struct {
	Success    bool    `json:"success"`
	Confidence float32 `json:"confidence"`
	Label      string  `json:"label"`
}

type FaceMatchResponse struct {
	Success    bool    `json:"success"`
	Similarity float32 `json:"similarity"`
}

type ModelRegisterResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type ModelDeleteResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type FaceDetectionPrediction struct {
	Confidence float32 `json:"confidence"`
	Ymin       int     `json:"y_min"`
	Xmin       int     `json:"x_min"`
	Ymax       int     `json:"y_max"`
	Xmax       int     `json:"x_max"`
}

type FaceRecognitionPrediction struct {
	Confidence float32 `json:"confidence"`
	Userid     string  `json:"userid"`
	Ymin       int     `json:"y_min"`
	Xmin       int     `json:"x_min"`
	Ymax       int     `json:"y_max"`
	Xmax       int     `json:"x_max"`
}

type FaceDeleteResponse struct {
	Success bool `json:"success"`
}

type FaceRegisterResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type FaceDetectionResponse struct {
	Success     bool                      `json:"success"`
	Predictions []FaceDetectionPrediction `json:"predictions"`
}

type FaceRecognitionResponse struct {
	Success     bool                        `json:"success"`
	Predictions []FaceRecognitionPrediction `json:"predictions"`
}

type FacelistResponse struct {
	Success bool     `json:"success"`
	Faces   []string `json:"faces"`
}

type DetectionPrediction struct {
	Confidence float32 `json:"confidence"`
	Label      string  `json:"label"`
	Ymin       int     `json:"y_min"`
	Xmin       int     `json:"x_min"`
	Ymax       int     `json:"y_max"`
	Xmax       int     `json:"x_max"`
}

type DetectionResponse struct {
	Success     bool                  `json:"success"`
	Predictions []DetectionPrediction `json:"predictions"`
}

type RestoreResponse struct {
	Success bool `json:"success"`
}

type ActivationSuccessResponse struct {
	Success      bool   `json:"success"`
	Expiringdate string `json:"expiringdate"`
	Daysleft     int    `json:"daysleft"`
	Plan         int    `json:"plan"`
}

type ActivationServerResponse struct {
	Responsecode        int    `json:"response_code"`
	Numberofactivations int    `json:"number_of_activations"`
	Plan                int    `json:"plan"`
	Datecreated         string `json:"date_created"`
	Dateexpiring        string `json:"date_expiring"`
}
