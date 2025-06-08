import React, { useEffect, useState } from "react";
import { View, Text, FlatList, Button } from "react-native";
import { getRebates } from "../services/api";

const RebateListScreen = ({ token }: { token: string }) => {
  const [rebates, setRebates] = useState<any[]>([]);
  const [error, setError] = useState("");

  const fetchRebates = async () => {
    setError("");
    try {
      const data = await getRebates(token);
      setRebates(data);
    } catch (e) {
      setError("Failed to fetch rebates");
    }
  };

  useEffect(() => {
    fetchRebates();
  }, []);

  return (
    <View>
      <Button title="Refresh" onPress={fetchRebates} />
      {error ? <Text style={{ color: "red" }}>{error}</Text> : null}
      <FlatList
        data={rebates}
        keyExtractor={(item) => item.id.toString()}
        renderItem={({ item }) => (
          <View>
            <Text>{item.title} - {item.amount}</Text>
          </View>
        )}
      />
    </View>
  );
};

export default RebateListScreen; 