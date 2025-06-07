package models

import (
	"golang.org/x/crypto/bcrypt"
)

type User struct {
	ID       int
	Username string
	Password string // hashed
}

var users = []User{}
var nextUserID = 1

func CreateUser(username, password string) (*User, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}
	user := User{
		ID:       nextUserID,
		Username: username,
		Password: string(hash),
	}
	nextUserID++
	users = append(users, user)
	return &user, nil
}

func GetUserByUsername(username string) *User {
	for _, u := range users {
		if u.Username == username {
			return &u
		}
	}
	return nil
}

func CheckPassword(user *User, password string) bool {
	return bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)) == nil
}
