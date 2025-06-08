import React from 'react';
import { View, Text, StyleSheet, ScrollView, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';

// Mock data for demonstration
const mockMaterial = {
  id: '1',
  name: 'Plastic Bottle',
  category: 'Plastic',
  points: 10,
  date: '2024-03-15',
  image: 'https://example.com/plastic-bottle.jpg',
  description: 'A clear plastic water bottle made from PET (Polyethylene Terephthalate).',
  recyclingInstructions: [
    'Remove the cap and label',
    'Rinse the bottle thoroughly',
    'Flatten the bottle to save space',
    'Place in the plastic recycling bin',
  ],
  environmentalImpact: {
    co2Saved: 0.5, // kg
    waterSaved: 2, // liters
    energySaved: 0.3, // kWh
  },
};

const MaterialDetailsScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        <View style={styles.header}>
          <View style={styles.categoryTag}>
            <MaterialIcons name="category" size={20} color="#4CAF50" />
            <Text style={styles.categoryText}>{mockMaterial.category}</Text>
          </View>
          <Text style={styles.title}>{mockMaterial.name}</Text>
          <Text style={styles.date}>Recycled on {mockMaterial.date}</Text>
        </View>

        <View style={styles.imageContainer}>
          <Image
            source={{ uri: mockMaterial.image }}
            style={styles.image}
            defaultSource={require('../assets/placeholder.png')}
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Description</Text>
          <Text style={styles.description}>{mockMaterial.description}</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recycling Instructions</Text>
          {mockMaterial.recyclingInstructions.map((instruction, index) => (
            <View key={index} style={styles.instructionItem}>
              <MaterialIcons name="check-circle" size={20} color="#4CAF50" />
              <Text style={styles.instructionText}>{instruction}</Text>
            </View>
          ))}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Environmental Impact</Text>
          <View style={styles.impactContainer}>
            <View style={styles.impactItem}>
              <MaterialIcons name="cloud" size={24} color="#4CAF50" />
              <Text style={styles.impactValue}>{mockMaterial.environmentalImpact.co2Saved} kg</Text>
              <Text style={styles.impactLabel}>CO₂ Saved</Text>
            </View>
            <View style={styles.impactItem}>
              <MaterialIcons name="water-drop" size={24} color="#4CAF50" />
              <Text style={styles.impactValue}>{mockMaterial.environmentalImpact.waterSaved} L</Text>
              <Text style={styles.impactLabel}>Water Saved</Text>
            </View>
            <View style={styles.impactItem}>
              <MaterialIcons name="bolt" size={24} color="#4CAF50" />
              <Text style={styles.impactValue}>{mockMaterial.environmentalImpact.energySaved} kWh</Text>
              <Text style={styles.impactLabel}>Energy Saved</Text>
            </View>
          </View>
        </View>

        <View style={styles.pointsContainer}>
          <MaterialIcons name="stars" size={24} color="#FFD700" />
          <Text style={styles.pointsText}>+{mockMaterial.points} points earned</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: 'white',
  },
  categoryTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
    alignSelf: 'flex-start',
    marginBottom: 10,
  },
  categoryText: {
    marginLeft: 5,
    color: '#4CAF50',
    fontSize: 14,
    fontWeight: 'bold',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  date: {
    fontSize: 14,
    color: '#666',
  },
  imageContainer: {
    height: 200,
    backgroundColor: '#E8F5E9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  section: {
    backgroundColor: 'white',
    padding: 20,
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  description: {
    fontSize: 16,
    color: '#666',
    lineHeight: 24,
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  instructionText: {
    fontSize: 16,
    color: '#666',
    marginLeft: 10,
    flex: 1,
  },
  impactContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  impactItem: {
    alignItems: 'center',
    flex: 1,
  },
  impactValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginTop: 5,
  },
  impactLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  pointsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFF9C4',
    padding: 15,
    margin: 20,
    borderRadius: 10,
  },
  pointsText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F57F17',
    marginLeft: 10,
  },
});

export default MaterialDetailsScreen; 