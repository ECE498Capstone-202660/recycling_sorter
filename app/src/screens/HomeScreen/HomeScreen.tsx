import React, { useState, useEffect, useMemo } from "react";
import { View, ScrollView, Image, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { MaterialIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { Card, Text, Button, Surface, Chip, IconButton } from "react-native-paper";
import styles from "./HomeScreen.styles";
import { useLatestResults } from "../../hooks/useLatestResults";
import { useAuth } from "../../context/AuthContext";
import { useHistorySummary } from "../../hooks/useHistorySummary";
import { acceptSorting, acknowledgeSorting, getSortingStatus } from "../../services/api";

type RecyclingStatus = "idle" | "awaiting_user" | "queued" | "processing" | "done" | "busy";

const HomeScreen = ({ navigation }: any) => {
  const { token, logout } = useAuth();
  const { results, loading } = useLatestResults(token);
  const { count: itemCount, totalRebate } = useHistorySummary(token);
  const [recyclingStatus, setRecyclingStatus] = useState<RecyclingStatus>("idle");
  const [statusLoading, setStatusLoading] = useState(true);
  const [jobId, setJobId] = useState<number | null>(null);

  const latest = useMemo(() => (results.length > 0 ? results[0] : null), [results]);

  useEffect(() => {
    let mounted = true;
    if (!token) {
      setRecyclingStatus("idle");
      setStatusLoading(false);
      setJobId(null);
      return;
    }

    const fetchStatus = async () => {
      try {
        const data = await getSortingStatus(token);
        if (!mounted) return;
        const nextStatus = (data?.status as RecyclingStatus) || "idle";
        setRecyclingStatus(nextStatus);
        setJobId(typeof data?.job_id === "number" ? data.job_id : null);
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
      subtitle: "Place an item in the slot to begin recycling.",
    },
    awaiting_user: {
      title: "Item detected",
      subtitle: "Tap to confirm and start recycling.",
    },
    queued: {
      title: "Queued",
      subtitle: "The machine is getting ready.",
    },
    processing: {
      title: "Recycling",
      subtitle: "Processing the current item now.",
    },
    done: {
      title: "Complete",
      subtitle: "Review the latest result.",
    },
    busy: {
      title: "Kiosk busy",
      subtitle: "Another user is recycling right now.",
    },
  };

  const statusToneMap: Record<RecyclingStatus, { bg: string; text: string }> = {
    idle: { bg: "#E3F1EE", text: "#0F6B6E" },
    awaiting_user: { bg: "#FFF0D9", text: "#8B5E00" },
    queued: { bg: "#E6EDF9", text: "#2A4C8B" },
    processing: { bg: "#DFF3E4", text: "#1D6C3B" },
    done: { bg: "#E9E4F8", text: "#4B2B8B" },
    busy: { bg: "#FBE2E2", text: "#8B2B2B" },
  };

  const statusLabel = statusLabelMap[recyclingStatus];
  const statusTone = statusToneMap[recyclingStatus];
  const showResult = recyclingStatus === "done";

  const handleAccept = async () => {
    if (!token || jobId == null) return;
    const res = await acceptSorting(token, jobId);
    if (res.ok) {
      setRecyclingStatus("queued");
    }
  };

  const handleFinish = async () => {
    if (token && jobId != null) {
      await acknowledgeSorting(token, jobId);
    }
    setRecyclingStatus("idle");
    setJobId(null);
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
      <Text style={styles.statusText}>
        Waiting for the kiosk to detect an item.
      </Text>
    );
  } else if (recyclingStatus === "awaiting_user") {
    controlContent = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusTitle}>Item detected</Text>
        <Text style={styles.statusText}>Start recycling this item?</Text>
        <Button mode="contained" onPress={handleAccept} style={styles.primaryButton}>
          Start recycling
        </Button>
      </View>
    );
  } else if (recyclingStatus === "queued" || recyclingStatus === "processing") {
    controlContent = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusTitle}>Recycling in progress</Text>
        <Text style={styles.statusText}>
          Please wait while the kiosk processes the item.
        </Text>
        <ActivityIndicator size="large" color="#0F6B6E" style={{ marginTop: 12 }} />
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
    controlContent = null;
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
              {showResult ? (
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
