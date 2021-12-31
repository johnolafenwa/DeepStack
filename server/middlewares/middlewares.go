package middlewares

import (
	"os"
	"strconv"

	"deepstack.io/server/response"
	"deepstack.io/server/structures"
	"deepstack.io/server/utils"
	"github.com/gin-gonic/gin"
)

func CheckFace() gin.HandlerFunc {

	return func(c *gin.Context) {

		activate_face := false
		face1 := os.Getenv("VISION-FACE")

		face2 := os.Getenv("VISION-FACE2")

		if face1 == "True" {

			activate_face = true

		}

		if face2 == "True" {

			activate_face = true

		}

		if activate_face == false {

			response := response.ErrorResponse{Success: false, Error: "Face endpoints not activated"}
			c.JSON(403, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}
}

func CheckScene() gin.HandlerFunc {

	return func(c *gin.Context) {

		activate_face := false
		scene := os.Getenv("VISION-SCENE")

		if scene == "True" {

			activate_face = true

		}

		if activate_face == false {

			response := response.ErrorResponse{Success: false, Error: "Scene endpoint not activated"}
			c.JSON(403, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}
}

func CheckCustomVision() gin.HandlerFunc {

	return func(c *gin.Context) {

		activate_custom_vision := false
		custom := os.Getenv("VISION-CUSTOM")

		if custom == "True" {

			activate_custom_vision = true

		}

		if activate_custom_vision == false {

			response := response.ErrorResponse{Success: false, Error: "Custom vision endpoint not activated"}
			c.JSON(403, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}
}

func CheckDetection() gin.HandlerFunc {

	return func(c *gin.Context) {

		activate_detection := false
		detection := os.Getenv("VISION-DETECTION")

		if detection == "True" {

			activate_detection = true

		}

		if activate_detection == false {

			response := response.ErrorResponse{Success: false, Error: "Detection endpoint not activated"}
			c.JSON(403, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}
}

func CheckSuperresolution() gin.HandlerFunc {

	return func(c *gin.Context) {

		activate_enhance := false
		enhance := os.Getenv("VISION-ENHANCE")

		if enhance == "True" {

			activate_enhance = true

		}

		if activate_enhance == false {

			response := response.ErrorResponse{Success: false, Error: "Enhance endpoint not activated"}
			c.JSON(403, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}
}

func CheckApiKey(sub_data *structures.ActivationData, settings *structures.Settings) gin.HandlerFunc {

	return func(c *gin.Context) {

		if settings.API_KEY != "" {

			passed_key := c.PostForm("api_key")

			if passed_key != settings.API_KEY {

				response := response.ErrorResponse{Success: false, Error: "Incorrect api key"}
				c.JSON(401, response)
				c.Abort()

			}

		}

	}
}

func CheckAdminKey(sub_data *structures.ActivationData, settings *structures.Settings) gin.HandlerFunc {

	return func(c *gin.Context) {

		if settings.ADMIN_KEY != "" {

			passed_key := c.PostForm("admin_key")

			if passed_key != settings.ADMIN_KEY {

				response := response.ErrorResponse{Success: false, Error: "Incorrect admin key"}
				c.JSON(401, response)
				c.Abort()

			}

		}

	}
}

func CheckPremium(sub_data *structures.ActivationData) gin.HandlerFunc {

	return func(c *gin.Context) {

		response := response.ErrorResponse{Success: false, Error: "Premium subscription required"}
		c.JSON(403, response)
		c.Abort()

	}
}

func CheckRegisterModel(sub_data *structures.ActivationData, DATA_DIR string) gin.HandlerFunc {

	return func(c *gin.Context) {

		_, err := c.FormFile("model")

		if err != nil {

			res := response.ErrorResponse{Success: false, Error: "No valid model file found"}

			c.JSON(403, res)
			c.Abort()

			return

		}

		_, err2 := c.FormFile("config")

		if err2 != nil {

			res := response.ErrorResponse{Success: false, Error: "No valid config file found"}

			c.JSON(403, res)
			c.Abort()

			return

		}

		model_name := c.PostForm("name")

		if model_name == "" {

			res := response.ErrorResponse{Success: false, Error: "Model name not specified"}

			c.JSON(403, res)
			c.Abort()

			return

		}

	}
}

func CheckImage() gin.HandlerFunc {

	return func(c *gin.Context) {

		_, err := c.FormFile("image")

		if err != nil {

			response := response.ErrorResponse{Success: false, Error: "No valid image file found"}

			c.JSON(400, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}

}

func CheckRestore() gin.HandlerFunc {

	return func(c *gin.Context) {

		_, err := c.FormFile("file")

		if err != nil {

			response := response.ErrorResponse{Success: false, Error: "No valid backup file found"}

			c.JSON(400, response)
			c.Abort()

			return
		} else {
			c.Next()
		}

	}

}

func CheckConfidence() gin.HandlerFunc {

	return func(c *gin.Context) {

		nms := c.PostForm("min_confidence")

		if nms != "" {
			val, err := strconv.ParseFloat(nms, 32)

			if err != nil {

				response := response.ErrorResponse{Success: false, Error: "Invalid value for min_confidence"}

				c.JSON(400, response)
				c.Abort()

				return

			} else {

				if val > 1.0 {

					response := response.ErrorResponse{Success: false, Error: "min_confidence cannot be greater than 1"}

					c.JSON(400, response)
					c.Abort()
					return
				}

			}

		}

	}
}

func CheckUserID() gin.HandlerFunc {

	return func(c *gin.Context) {

		userid := c.PostForm("userid")

		if userid == "" {

			response := response.ErrorResponse{Success: false, Error: "userid not specified"}

			c.JSON(400, response)
			c.Abort()
			return

		}

	}
}

func CheckDeleteModel(DATA_DIR string) gin.HandlerFunc {

	return func(c *gin.Context) {

		model_name := c.PostForm("name")

		if model_name == "" {

			response := response.ErrorResponse{Success: false, Error: "Model name not specified"}

			c.JSON(400, response)
			c.Abort()
			return

		} else {

			MODEL_DIR := DATA_DIR + "/models/vision/" + model_name + "/"

			model_exists, _ := utils.PathExists(MODEL_DIR)

			if model_exists == false {

				response := response.ErrorResponse{Success: false, Error: "Model does not exist"}

				c.JSON(400, response)
				c.Abort()

				return

			}

		}

	}
}

func CheckMultiImage() gin.HandlerFunc {

	return func(c *gin.Context) {

		form, _ := c.MultipartForm()

		count := len(form.File)

		if count == 0 {

			response := response.ErrorResponse{Success: false, Error: "No valid image file found"}

			c.JSON(400, response)
			c.Abort()
		} else {
			c.Next()
		}

	}
}

func CheckFaceMatch() gin.HandlerFunc {

	return func(c *gin.Context) {

		form, _ := c.MultipartForm()

		count := len(form.File)

		if count == 0 {

			response := response.ErrorResponse{Success: false, Error: "No valid image file found"}

			c.JSON(400, response)
			c.Abort()
		} else if count == 1 {

			response := response.ErrorResponse{Success: false, Error: "expected two image files, only one found"}

			c.JSON(400, response)
			c.Abort()

		} else if count > 2 {

			response := response.ErrorResponse{Success: false, Error: "only two images allowed"}

			c.JSON(400, response)
			c.Abort()

		}

	}
}
