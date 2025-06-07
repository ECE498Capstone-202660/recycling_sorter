package main

import (
	"backend/handlers"
	"backend/middleware"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	r.Use(cors.Default())
	
	r.POST("/register", handlers.Register)
	r.POST("/login", handlers.Login)

	auth := r.Group("/")
	auth.Use(middleware.JWTAuth())
	{
		auth.GET("/rebates", handlers.ListRebates)
		auth.POST("/rebates", handlers.CreateRebate)
		auth.GET("/rebates/:id", handlers.GetRebate)
		auth.PUT("/rebates/:id", handlers.UpdateRebate)
		auth.DELETE("/rebates/:id", handlers.DeleteRebate)
	}

	r.Run() // listen and serve on 0.0.0.0:8080
}
