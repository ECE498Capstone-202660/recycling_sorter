import React, { useState, useEffect } from "react";
import { View, Text, ScrollView, TouchableOpacity, Image, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { MaterialIcons } from "@expo/vector-icons";
import styles from "./HomeScreen.styles";
import { useLatestResults } from "../../hooks/useLatestResults";

const HomeScreen = ({ navigation }: any) => {
  const { results, loading } = useLatestResults();
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    if (current >= results.length) {
      setCurrent(0);
    }
  }, [results.length, current]);

  const hasResults = results.length > 0;
  const currentResult = hasResults ? results[current] : null;
  const canFlip = results.length > 1;

  const handleFlip = (direction: 'left' | 'right') => {
    setCurrent((prev) => {
      if (direction === 'left') {
        return prev === 0 ? results.length - 1 : prev - 1;
      } else {
        return prev === results.length - 1 ? 0 : prev + 1;
      }
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        <View style={styles.header}>
          <Text style={styles.title}>Recycling Sorter</Text>
          <Text style={styles.subtitle}>Make recycling easier</Text>
        </View>

        <View style={styles.uploadContainer}>
          <Text style={styles.sectionTitle}>Top 3 Latest Classifications</Text>
          {loading ? (
            <ActivityIndicator size="large" color="#4CAF50" style={{ marginTop: 10 }} />
          ) : !hasResults ? (
            <Text style={{ color: '#888', marginTop: 20 }}>No image classified yet.</Text>
          ) : (
            <View>
              <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
                {canFlip && (
                  <TouchableOpacity onPress={() => handleFlip('left')} style={{ padding: 8 }}>
                    <MaterialIcons name="chevron-left" size={36} color="#4CAF50" />
                  </TouchableOpacity>
                )}
                <Image
                  source={{ uri: currentResult.image_url }}
                  style={styles.previewImage}
                />
                {canFlip && (
                  <TouchableOpacity onPress={() => handleFlip('right')} style={{ padding: 8 }}>
                    <MaterialIcons name="chevron-right" size={36} color="#4CAF50" />
                  </TouchableOpacity>
                )}
              </View>
              <View style={styles.resultContainer}>
                <Text style={styles.resultTitle}>Classification Result</Text>
                <Text style={styles.resultText}>Material: <Text style={styles.resultValue}>{currentResult.predicted_class}</Text></Text>
                <Text style={styles.resultText}>Rebate: <Text style={styles.resultValue}>${currentResult.rebate?.toFixed(2) ?? '--'}</Text></Text>
                <Text style={styles.resultText}>Confidence: <Text style={styles.resultValue}>{currentResult.confidence ?? '--'}</Text></Text>
                <View style={styles.feedbackRow}>
                  <TouchableOpacity style={styles.feedbackButton}>
                    <MaterialIcons name="thumb-up" size={28} color="#4CAF50" />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.feedbackButton}>
                    <MaterialIcons name="thumb-down" size={28} color="#f44336" />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          )}
        </View>

        <View style={styles.statsContainer}>
          <View style={styles.statCard}>
            <MaterialIcons name="recycling" size={32} color="#4CAF50" />
            <Text style={styles.statNumber}>24</Text>
            <Text style={styles.statLabel}>Items Recycled</Text>
          </View>
          <View style={styles.statCard}>
            <MaterialIcons name="eco" size={32} color="#4CAF50" />
            <Text style={styles.statNumber}>12.5</Text>
            <Text style={styles.statLabel}>kg CO 2 Saved</Text>
          </View>
        </View>

        <View style={styles.quickActions}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <TouchableOpacity style={styles.actionButton} onPress={() => navigation.navigate('History')}>
            <MaterialIcons name="history" size={24} color="white" />
            <Text style={styles.actionButtonText}>View History</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.tipsContainer}>
          <Text style={styles.sectionTitle}>Recycling Tips</Text>
          <View style={styles.tipCard}>
            <MaterialIcons name="lightbulb" size={24} color="#4CAF50" />
            <Text style={styles.tipText}>
              Rinse containers before recycling to prevent contamination
            </Text>
          </View>
          <View style={styles.tipCard}>
            <MaterialIcons name="lightbulb" size={24} color="#4CAF50" />
            <Text style={styles.tipText}>
              Check local recycling guidelines for specific material acceptance
            </Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default HomeScreen; 