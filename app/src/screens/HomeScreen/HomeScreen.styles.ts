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
    paddingTop: 8,
    paddingHorizontal: 20,
    paddingBottom: 24,
  },
  heroAccent: {
    position: 'absolute',
    top: -40,
    right: -50,
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: '#CFE7E2',
    opacity: 0.6,
  },
  heroContent: {
    gap: 12,
  },
  heroTitle: {
    color: '#0B2F33',
  },
  heroSubtitle: {
    color: '#385457',
  },
  heroStats: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 8,
  },
  heroStatCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 16,
    backgroundColor: '#FFFFFF',
  },
  heroStatValue: {
    fontSize: 18,
    color: '#0F6B6E',
    fontWeight: '600',
  },
  heroStatLabel: {
    fontSize: 12,
    color: '#577375',
  },
  sectionCard: {
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 20,
    backgroundColor: '#FFFFFF',
  },
  sectionHeader: {
    marginBottom: 12,
  },
  sectionTitle: {
    color: '#0B2F33',
  },
  sectionSubtitle: {
    color: '#617779',
    marginTop: 4,
  },
  emptyState: {
    color: '#7C8B8C',
    marginTop: 12,
  },
  singleImageWrap: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  previewImage: {
    width: 300,
    height: 300,
    borderRadius: 22,
    resizeMode: 'cover',
    backgroundColor: '#E6F1EE',
  },
  resultPanel: {
    marginTop: 12,
    padding: 14,
    borderRadius: 16,
    backgroundColor: '#F2F6F4',
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0F6B6E',
    marginBottom: 8,
  },
  resultWarning: {
    color: '#8B2B2B',
    marginBottom: 8,
  },
  chipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  infoChip: {
    backgroundColor: '#FFFFFF',
  },
  resultActions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
    justifyContent: 'flex-start',
  },
  centerBlock: {
    alignItems: 'center',
    gap: 10,
  },
  statusTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0B2F33',
  },
  statusText: {
    fontSize: 14,
    color: '#567177',
    textAlign: 'center',
  },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  primaryButton: {
    borderRadius: 14,
  },
  primaryButtonContent: {
    paddingVertical: 6,
  },
  secondaryButton: {
    borderRadius: 14,
    borderColor: '#0F6B6E',
  },
  recyclingCard: {
    borderRadius: 18,
    backgroundColor: '#F0F6F4',
    padding: 24,
    gap: 14,
    minHeight: 480,
    justifyContent: 'space-between',
  },
  recyclingCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  statusPill: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  statusPillText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  resultInline: {
    marginTop: 16,
  },
  recyclingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0B2F33',
  },
  recyclingSubtitle: {
    marginTop: 6,
    color: '#5C7375',
    maxWidth: 240,
  },
  tipRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 12,
  },
  tipText: {
    flex: 1,
    color: '#5A6E70',
  },
});

export default styles;
