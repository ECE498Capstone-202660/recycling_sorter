import React from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';

// Mock data for demonstration
const mockHistory = [
  {
    id: '1',
    material: 'Plastic Bottle',
    date: '2024-03-15',
    category: 'Plastic',
    points: 10,
  },
  {
    id: '2',
    material: 'Aluminum Can',
    date: '2024-03-14',
    category: 'Metal',
    points: 5,
  },
  {
    id: '3',
    material: 'Cardboard Box',
    date: '2024-03-13',
    category: 'Paper',
    points: 8,
  },
];

const HistoryScreen = () => {
  const renderItem = ({ item }: { item: typeof mockHistory[0] }) => (
    <TouchableOpacity style={styles.historyItem}>
      <View style={styles.itemHeader}>
        <Text style={styles.materialName}>{item.material}</Text>
        <Text style={styles.points}>+{item.points} pts</Text>
      </View>
      <View style={styles.itemDetails}>
        <View style={styles.categoryTag}>
          <MaterialIcons name="category" size={16} color="#4CAF50" />
          <Text style={styles.categoryText}>{item.category}</Text>
        </View>
        <Text style={styles.date}>{item.date}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Recycling History</Text>
        <View style={styles.statsContainer}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>23</Text>
            <Text style={styles.statLabel}>Total Items</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>156</Text>
            <Text style={styles.statLabel}>Points Earned</Text>
          </View>
        </View>
      </View>

      <FlatList
        data={mockHistory}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContainer}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#4CAF50',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 15,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  statLabel: {
    fontSize: 14,
    color: 'white',
    opacity: 0.8,
  },
  listContainer: {
    padding: 15,
  },
  historyItem: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  materialName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  points: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  itemDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  categoryTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  categoryText: {
    marginLeft: 5,
    color: '#4CAF50',
    fontSize: 12,
  },
  date: {
    fontSize: 12,
    color: '#666',
  },
});

export default HistoryScreen; 