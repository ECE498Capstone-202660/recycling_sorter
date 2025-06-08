package handlers

import (
	"net/http"
	"backend/models"
	"strconv"

	"github.com/gin-gonic/gin"
)

type RebateRequest struct {
	Title  string `json:"title"`
	Amount int    `json:"amount"`
}

func getUserID(c *gin.Context) int {
	userIDStr, _ := c.Get("user_id")
	userID, _ := strconv.Atoi(userIDStr.(string))
	return userID
}

func ListRebates(c *gin.Context) {
	userID := getUserID(c)
	rebates := models.ListRebates(userID)
	c.JSON(http.StatusOK, rebates)
}

func CreateRebate(c *gin.Context) {
	userID := getUserID(c)
	var req RebateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	rebate := models.CreateRebate(userID, req.Title, req.Amount)
	c.JSON(http.StatusOK, rebate)
}

func GetRebate(c *gin.Context) {
	userID := getUserID(c)
	id, _ := strconv.Atoi(c.Param("id"))
	rebate := models.GetRebateByID(userID, id)
	if rebate == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Rebate not found"})
		return
	}
	c.JSON(http.StatusOK, rebate)
}

func UpdateRebate(c *gin.Context) {
	userID := getUserID(c)
	id, _ := strconv.Atoi(c.Param("id"))
	var req RebateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	rebate := models.UpdateRebate(userID, id, req.Title, req.Amount)
	if rebate == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Rebate not found"})
		return
	}
	c.JSON(http.StatusOK, rebate)
}

func DeleteRebate(c *gin.Context) {
	userID := getUserID(c)
	id, _ := strconv.Atoi(c.Param("id"))
	if !models.DeleteRebate(userID, id) {
		c.JSON(http.StatusNotFound, gin.H{"error": "Rebate not found"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "Rebate deleted"})
}
