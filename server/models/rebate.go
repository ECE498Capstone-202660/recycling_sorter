package models

type Rebate struct {
	ID     int    `json:"id"`
	UserID int    `json:"user_id"`
	Title  string `json:"title"`
	Amount int    `json:"amount"`
}

var rebates = []Rebate{}
var nextRebateID = 1

func CreateRebate(userID int, title string, amount int) *Rebate {
	rebate := Rebate{
		ID:     nextRebateID,
		UserID: userID,
		Title:  title,
		Amount: amount,
	}
	nextRebateID++
	rebates = append(rebates, rebate)
	return &rebate
}

func ListRebates(userID int) []Rebate {
	var result []Rebate
	for _, r := range rebates {
		if r.UserID == userID {
			result = append(result, r)
		}
	}
	return result
}

func GetRebateByID(userID, id int) *Rebate {
	for _, r := range rebates {
		if r.ID == id && r.UserID == userID {
			return &r
		}
	}
	return nil
}

func UpdateRebate(userID, id int, title string, amount int) *Rebate {
	for i, r := range rebates {
		if r.ID == id && r.UserID == userID {
			rebates[i].Title = title
			rebates[i].Amount = amount
			return &rebates[i]
		}
	}
	return nil
}

func DeleteRebate(userID, id int) bool {
	for i, r := range rebates {
		if r.ID == id && r.UserID == userID {
			rebates = append(rebates[:i], rebates[i+1:]...)
			return true
		}
	}
	return false
}
