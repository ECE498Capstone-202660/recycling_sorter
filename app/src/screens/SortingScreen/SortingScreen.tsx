import React, { useEffect, useMemo, useState } from "react";
import { View, Image } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { ActivityIndicator, Button, Card, Text } from "react-native-paper";
import styles from "./SortingScreen.styles";
import { useAuth } from "../../context/AuthContext";
import { acceptSorting, acknowledgeSorting, getSortingStatus } from "../../services/api";
import { useLatestResults } from "../../hooks/useLatestResults";

type Status = "idle" | "awaiting_user" | "queued" | "processing" | "done" | "busy";

const SortingScreen = () => {
  const { token, logout } = useAuth();
  const { results, loading: latestLoading } = useLatestResults(token);
  const latest = useMemo(() => (results.length > 0 ? results[0] : null), [results]);
  const [status, setStatus] = useState<Status>("idle");
  const [jobId, setJobId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    if (!token) {
      setStatus("idle");
      setJobId(null);
      setLoading(false);
      return;
    }

    const fetchStatus = async () => {
      try {
        const data = await getSortingStatus(token);
        if (!mounted) return;
        setStatus((data?.status as Status) || "idle");
        setJobId(typeof data?.job_id === "number" ? data.job_id : null);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchStatus();
    const id = setInterval(fetchStatus, 2000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [token]);

  const handleAccept = async () => {
    if (!token || jobId == null) return;
    const res = await acceptSorting(token, jobId);
    if (res.ok) {
      setStatus("queued");
    }
  };

  const handleContinue = async () => {
    if (token && jobId != null) {
      await acknowledgeSorting(token, jobId);
    }
    setStatus("idle");
    setJobId(null);
  };

  const handleFinish = async () => {
    if (token && jobId != null) {
      await acknowledgeSorting(token, jobId);
    }
    logout();
  };

  let content: React.ReactNode = null;
  if (loading) {
    content = <ActivityIndicator size="large" color="#0F6B6E" />;
  } else if (status === "idle") {
    content = (
      <Text style={styles.statusText}>
        Waiting for the machine to detect an item.
      </Text>
    );
  } else if (status === "awaiting_user") {
    content = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusTitle}>Item detected</Text>
        <Text style={styles.statusText}>
          Start recycling this item?
        </Text>
        <Button mode="contained" onPress={handleAccept} style={styles.primaryButton}>
          Start recycling
        </Button>
      </View>
    );
  } else if (status === "queued" || status === "processing") {
    content = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusTitle}>Recycling in progress</Text>
        <Text style={styles.statusText}>
          Please wait while the machine processes the item.
        </Text>
        <ActivityIndicator size="large" color="#0F6B6E" style={{ marginTop: 12 }} />
      </View>
    );
  } else if (status === "busy") {
    content = (
      <View style={styles.centerBlock}>
        <Text style={styles.statusTitle}>Machine busy</Text>
        <Text style={styles.statusText}>
          Another user is currently sorting. Please wait.
        </Text>
      </View>
    );
  } else if (status === "done") {
    content = (
      <View>
        <Text style={styles.statusTitle}>Recycling complete</Text>
        <Text style={styles.statusText}>Here is your latest result.</Text>
        {latest && !latestLoading ? (
          <Card style={styles.resultCard}>
            <Card.Content style={styles.resultContent}>
              <Image source={{ uri: latest.image_url }} style={styles.previewImage} />
              <View style={styles.resultDetails}>
                <Text style={styles.resultLabel}>{latest.predicted_class}</Text>
                <Text style={styles.resultMeta}>${latest.rebate?.toFixed(2) ?? "--"}</Text>
                <Text style={styles.resultMeta}>{latest.confidence ?? "--"}</Text>
              </View>
            </Card.Content>
          </Card>
        ) : (
          <ActivityIndicator size="small" color="#0F6B6E" />
        )}
        <View style={styles.actionRow}>
          <Button mode="contained" onPress={handleContinue} style={styles.primaryButton}>
            Continue
          </Button>
          <Button mode="outlined" onPress={handleFinish} style={styles.secondaryButton}>
            Finish
          </Button>
        </View>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={["#E2F2F1", "#F6F7F4"]} style={styles.hero}>
        <Text variant="headlineMedium" style={styles.title}>
          Recycling Control
        </Text>
        <Text style={styles.subtitle}>Manage the current recycling session.</Text>
      </LinearGradient>
      <View style={styles.content}>
        {content}
      </View>
    </SafeAreaView>
  );
};

export default SortingScreen;
