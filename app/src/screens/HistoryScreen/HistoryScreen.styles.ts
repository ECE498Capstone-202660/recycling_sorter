import { StyleSheet } from "react-native";

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f2f4f5",
  },

  listContainer: {
    padding: 16,
  },

  historyItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#E8F5E9",   // <<< light green, matches Home screen
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },

  thumb: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 14,
  },

  itemContent: {
    flex: 1,
  },
  categoryText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#2e7d32",
  },
  date: {
    fontSize: 12,
    color: "#4f4f4f",
    marginTop: 2,
  },

  rebateBox: {
    flexDirection: "row",
    alignItems: "center",
  },
  rebateText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#2e7d32",
    marginLeft: 2,
  },
});

export default styles;
