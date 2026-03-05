import React, { useState, useEffect, useMemo } from "react";
import { View, ScrollView, Image, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { MaterialIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { Card, Text, Button, Surface, Chip } from "react-native-paper";
import styles from "./HomeScreen.styles";
import { useLatestResults } from "../../hooks/useLatestResults";
import { useAuth } from "../../context/AuthContext";
import { useHistorySummary } from "../../hooks/useHistorySummary";
import { getSortingStatus, startRecycling, stopRecycling } from "../../services/api";

type RecyclingStatus = "idle" | "active" | "busy";

const HomeScreen = ({ navigation }: any) => {
  const { token, logout } = useAuth();
  const { results, loading } = useLatestResults(token);
  const { count: itemCount, totalRebate } = useHistorySummary(token);
  const [recyclingStatus, setRecyclingStatus] = useState<RecyclingStatus>("idle");
  const [statusLoading, setStatusLoading] = useState(true);

  const latest = useMemo(() => (results.length > 0 ? results[0] : null), [results]);

  useEffect(() => {
    let mounted = true;
    if (!token) {
      setRecyclingStatus("idle");
      setStatusLoading(false);
      return;
    }

    const fetchStatus = async () => {
      try {
        const data = await getSortingStatus(token);
        if (!mounted) return;
        const nextStatus = (data?.status as RecyclingStatus) || "idle";
        setRecyclingStatus(nextStatus);
      } finally {
        if (mounted) setStatusLoading(false);
      }
    };

    fetchStatus();
    const id = setInterval(fetchStatus, 2000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [token]);

  const statusLabelMap: Record<RecyclingStatus, { title: string; subtitle: string }> = {
    idle: {
      title: "Kiosk idle",
      subtitle: "Tap start to begin a recycling session.",
    },
    active: {
      title: "Session active",
      subtitle: "Kiosk is ready to process items.",
    },
    busy: {
      title: "Kiosk busy",
      subtitle: "Another user is recycling right now.",
    },
  };

  const statusToneMap: Record<RecyclingStatus, { bg: string; text: string }> = {
    idle: { bg: "#E3F1EE", text: "#0F6B6E" },
    active: { bg: "#DFF3E4", text: "#1D6C3B" },
    busy: { bg: "#FBE2E2", text: "#8B2B2B" },
  };

  const statusLabel = statusLabelMap[recyclingStatus];
  const statusTone = statusToneMap[recyclingStatus];
  const showResult = recyclingStatus === "active";
  const hasSessionResult = showResult && !!latest;

  const handleStart = async () => {
    if (!token) return;
    const res = await startRecycling(token);
    if (res.ok) {
      setRecyclingStatus("active");
    } else if (res.status === 423) {
      setRecyclingStatus("busy");
    }
  };

  const handleFinish = async () => {
    if (token) {
      await stopRecycling(token);
    }
    setRecyclingStatus("idle");
  };

  const isTrash = latest?.predicted_class === "Trash";
  let latestContent: React.ReactNode;
  if (loading) {
    latestContent = <ActivityIndicator size="large" color="#0F6B6E" style={{ marginTop: 16 }} />;
  } else if (latest) {
    latestContent = (
      <View>
        <View style={styles.singleImageWrap}>
          <Image source={{ uri: latest.image_url }} style={styles.previewImage} />
        </View>
        <View style={styles.resultPanel}>
          <Text style={styles.resultTitle}>Classification result</Text>
          {isTrash ? (
            <Text style={styles.resultWarning}>
              This item is not recyclable. Please take it back.
            </Text>
          ) : null}
          <View style={styles.chipRow}>
            <Chip icon="recycle" style={styles.infoChip}>
              {latest.predicted_class}
            </Chip>
            <Chip icon="cash" style={styles.infoChip}>
              ${latest.rebate?.toFixed(2) ?? "--"}
            </Chip>
            <Chip icon="signal" style={styles.infoChip}>
              {latest.confidence ?? "--"}
            </Chip>
          </View>
          <View style={styles.resultActions}>
            <Button mode="outlined" onPress={handleFinish} style={styles.secondaryButton}>
              Finish
            </Button>
          </View>
        </View>
      </View>
    );
  } else {
    latestContent = <Text style={styles.emptyState}>No image classified yet.</Text>;
  }

  let controlContent: React.ReactNode;
  if (statusLoading) {
    controlContent = <ActivityIndicator size="small" color="#0F6B6E" />;
  } else if (recyclingStatus === "idle") {
    controlContent = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusText}>
          Tap start to begin a recycling session.
        </Text>
        <Button mode="contained" onPress={handleStart} style={styles.primaryButton}>
          Start recycling session
        </Button>
      </View>
    );
  } else if (recyclingStatus === "busy") {
    controlContent = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusTitle}>Kiosk busy</Text>
        <Text style={styles.statusText}>
          Another user is currently recycling. Please wait.
        </Text>
      </View>
    );
  } else {
    controlContent = (
      <View style={styles.centerBlock}>
        {!hasSessionResult ? (
          <>
            <ActivityIndicator size="small" color="#0F6B6E" />
            <Text style={styles.statusText}>Waiting for the kiosk result...</Text>
          </>
        ) : (
          <Text style={styles.statusText}>Listening for new results...</Text>
        )}
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <LinearGradient colors={["#E2F2F1", "#F6F7F4"]} style={styles.hero}>
          <View style={styles.heroAccent} />
          <View style={styles.heroContent}>
            <Text variant="headlineMedium" style={styles.heroTitle}>
              Recycling Station
            </Text>
            <Text variant="bodyMedium" style={styles.heroSubtitle}>
              Make recycling feel effortless with quick, reliable results.
            </Text>
            <View style={styles.heroStats}>
              <Surface style={styles.heroStatCard} elevation={0}>
                <MaterialIcons name="recycling" size={22} color="#0F6B6E" />
                <View>
                  <Text style={styles.heroStatValue}>{itemCount}</Text>
                  <Text style={styles.heroStatLabel}>Items recycled</Text>
                </View>
              </Surface>
              <Surface style={styles.heroStatCard} elevation={0}>
                <MaterialIcons name="savings" size={22} color="#0F6B6E" />
                <View>
                  <Text style={styles.heroStatValue}>${totalRebate.toFixed(2)}</Text>
                  <Text style={styles.heroStatLabel}>Rebate earned</Text>
                </View>
              </Surface>
            </View>
          </View>
        </LinearGradient>

        <Card style={styles.sectionCard}>
          <Card.Content>
            <Surface style={styles.recyclingCard} elevation={0}>
              <View style={styles.recyclingCardHeader}>
                <View>
                  <Text style={styles.recyclingTitle}>{statusLabel.title}</Text>
                  <Text style={styles.recyclingSubtitle}>{statusLabel.subtitle}</Text>
                </View>
                <View style={[styles.statusPill, { backgroundColor: statusTone.bg }]}>
                  <Text style={[styles.statusPillText, { color: statusTone.text }]}>
                    {recyclingStatus.replace("_", " ")}
                  </Text>
                </View>
              </View>
              {controlContent}
              {hasSessionResult ? (
                <View style={styles.resultInline}>
                  {latestContent}
                </View>
              ) : null}
            </Surface>
          </Card.Content>
        </Card>

        <Card style={styles.sectionCard}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Text variant="titleMedium" style={styles.sectionTitle}>
                Recycling tips
              </Text>
              <Text style={styles.sectionSubtitle}>Small habits, bigger impact</Text>
            </View>
            <View style={styles.tipRow}>
              <MaterialIcons name="lightbulb" size={20} color="#0F6B6E" />
              <Text style={styles.tipText}>
                Rinse containers before recycling to prevent contamination.
              </Text>
            </View>
            <View style={styles.tipRow}>
              <MaterialIcons name="lightbulb" size={20} color="#0F6B6E" />
              <Text style={styles.tipText}>
                Check local guidelines to confirm what materials are accepted.
              </Text>
            </View>
          </Card.Content>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
};

export default HomeScreen; 
