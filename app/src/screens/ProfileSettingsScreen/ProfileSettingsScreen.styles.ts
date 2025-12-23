import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F6F7F4',
  },
  scrollContent: {
    paddingBottom: 32,
    paddingTop: 12,
  },
  headerCard: {
    marginHorizontal: 16,
    borderRadius: 18,
    backgroundColor: '#FFFFFF',
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#E2F2F1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    color: '#0B2F33',
  },
  headerSubtitle: {
    color: '#5A6E70',
    marginTop: 2,
  },
  settingsCard: {
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 18,
    backgroundColor: '#FFFFFF',
  },
  sectionTitle: {
    color: '#0B2F33',
    marginBottom: 8,
  },
  input: {
    marginTop: 12,
    backgroundColor: '#F6F7F4',
  },
  saveButton: {
    marginTop: 16,
    borderRadius: 12,
    backgroundColor: '#0F6B6E',
  },
  successText: {
    marginTop: 12,
    color: '#0F6B6E',
    fontWeight: '600',
  },
  loadingIndicator: {
    marginTop: 12,
  },
});

export default styles;
