import React, { useEffect, useState } from "react";
import { View, Text, FlatList, TouchableOpacity, Image } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { MaterialIcons } from "@expo/vector-icons";
import styles from "./HistoryScreen.styles";
import { getHistory } from "../../services/api";

const HistoryScreen = () => {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    getHistory().then(setHistory);
  }, []);

  const renderItem = ({ item }: { item: any }) => (
    <TouchableOpacity style={styles.historyItem}>
      <Image source={{ uri: item.image_url }} style={styles.thumb} />
      <View style={styles.itemContent}>
        <Text style={styles.categoryText}>{item.category}</Text>
        <Text style={styles.date}>{item.date}</Text>
      </View>
      <View style={styles.rebateBox}>
        <MaterialIcons name="attach-money" size={18} color="#4CAF50" />
        <Text style={styles.rebateText}>{item.rebate.toFixed(2)}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <FlatList
        data={history}
        renderItem={renderItem}
        keyExtractor={(item) => item.id.toString()}
        contentContainerStyle={styles.listContainer}
      />
    </SafeAreaView>
  );
};

export default HistoryScreen;
