import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 32,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  hero: {
    marginBottom: 24,
    alignItems: 'center',
    maxWidth: 520,
  },
  title: {
    color: '#0B2F33',
  },
  subtitle: {
    color: '#4F6668',
    marginTop: 8,
    textAlign: 'center',
  },
  formCard: {
    borderRadius: 20,
    backgroundColor: '#FFFFFF',
    width: '100%',
    maxWidth: 420,
    alignSelf: 'center',
  },
  input: {
    marginBottom: 14,
  },
  inputOutline: {
    borderRadius: 12,
    borderColor: '#C7D1CC',
  },
  inputContent: {
    paddingLeft: 0,
  },
  button: {
    marginTop: 8,
    borderRadius: 14,
  },
  secondaryButton: {
    marginTop: 6,
  },
  error: {
    color: "#E45B52",
    marginBottom: 10,
    textAlign: 'center',
  },
});

export default styles; 
