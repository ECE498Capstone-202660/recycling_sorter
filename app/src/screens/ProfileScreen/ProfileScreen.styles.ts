import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F6F7F4',
  },
  scrollContent: {
    paddingBottom: 32,
  },
  hero: {
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 24,
  },
  heroContent: {
    alignItems: 'center',
    gap: 8,
  },
  profileImage: {
    width: 86,
    height: 86,
    borderRadius: 43,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  name: {
    color: '#0B2F33',
  },
  email: {
    color: '#5A6E70',
  },
  statsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginHorizontal: 16,
    marginTop: 8,
  },
  statCard: {
    flex: 1,
    minWidth: 110,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    paddingVertical: 12,
    paddingHorizontal: 14,
    alignItems: 'flex-start',
  },
  statValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0F6B6E',
  },
  statLabel: {
    fontSize: 12,
    color: '#6B7F80',
    marginTop: 4,
  },
  menuCard: {
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 18,
    backgroundColor: '#FFFFFF',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 14,
  },
  menuDivider: {
    borderBottomWidth: 1,
    borderBottomColor: '#EEF1EF',
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  menuItemText: {
    fontSize: 15,
    color: '#0B2F33',
  },
  logoutButton: {
    marginHorizontal: 16,
    marginTop: 20,
    borderRadius: 14,
    backgroundColor: '#0F6B6E',
  },
  logoutContent: {
    paddingVertical: 6,
  },
  loadingIndicator: {
    marginTop: 12,
  },
});

export default styles;
