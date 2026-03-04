import { StyleSheet } from "react-native";

export default StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F6F7F4",
  },
  hero: {
    paddingHorizontal: 20,
    paddingVertical: 24,
  },
  title: {
    color: "#0B2F33",
    fontWeight: "700",
  },
  subtitle: {
    color: "#567177",
    marginTop: 6,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  centerBlock: {
    alignItems: "center",
    gap: 10,
  },
  statusTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#0B2F33",
  },
  statusText: {
    fontSize: 14,
    color: "#567177",
    textAlign: "center",
  },
  primaryButton: {
    marginTop: 12,
    borderRadius: 12,
  },
  secondaryButton: {
    marginTop: 12,
    borderRadius: 12,
  },
  resultCard: {
    marginTop: 16,
    borderRadius: 16,
    overflow: "hidden",
  },
  resultContent: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
  },
  previewImage: {
    width: 96,
    height: 96,
    borderRadius: 12,
    backgroundColor: "#E6ECEB",
  },
  resultDetails: {
    flex: 1,
  },
  resultLabel: {
    fontSize: 18,
    fontWeight: "700",
    color: "#0B2F33",
  },
  resultMeta: {
    color: "#567177",
    marginTop: 4,
  },
  actionRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 16,
  },
});
