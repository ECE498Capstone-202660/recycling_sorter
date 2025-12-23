import React, { useEffect, useMemo, useState } from 'react';
import { View, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { ActivityIndicator, Button, Card, Text, TextInput } from 'react-native-paper';
import styles from './ProfileSettingsScreen.styles';
import { useAuth } from '../../context/AuthContext';
import { getUserMe, updateUserMe } from '../../services/api';

const ProfileSettingsScreen = () => {
  const { token } = useAuth();
  const [profile, setProfile] = useState({
    username: '',
    email: '',
    first_name: '',
    last_name: '',
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const displayName = useMemo(() => {
    const name = [profile.first_name, profile.last_name].filter(Boolean).join(' ');
    return name || profile.username || 'Your profile';
  }, [profile.first_name, profile.last_name, profile.username]);

  useEffect(() => {
    const loadProfile = async () => {
      if (!token) {
        setLoading(false);
        return;
      }
      setSuccessMessage('');
      setLoading(true);
      try {
        const data = await getUserMe(token);
        setProfile({
          username: data?.username ?? '',
          email: data?.email ?? '',
          first_name: data?.first_name ?? '',
          last_name: data?.last_name ?? '',
        });
      } catch (err) {
        console.error('Failed to load profile details.', err);
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [token]);

  const handleSave = async () => {
    if (!token) {
      setSuccessMessage('');
      return;
    }
    setSaving(true);
    setSuccessMessage('');
    const normalize = (value: string) => {
      const trimmed = value.trim();
      return trimmed.length ? trimmed : null;
    };
    try {
      const updated = await updateUserMe(token, {
        email: normalize(profile.email),
        first_name: normalize(profile.first_name),
        last_name: normalize(profile.last_name),
      });
      setProfile({
        username: updated?.username ?? profile.username,
        email: updated?.email ?? '',
        first_name: updated?.first_name ?? '',
        last_name: updated?.last_name ?? '',
      });
      setSuccessMessage('Saved successfully.');
    } catch (err) {
      console.error('Failed to save profile changes.', err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Card style={styles.headerCard}>
          <Card.Content>
            <View style={styles.headerRow}>
              <View style={styles.avatar}>
                <MaterialIcons name="person" size={28} color="#0F6B6E" />
              </View>
              <View>
                <Text variant="titleMedium" style={styles.headerTitle}>
                  {displayName}
                </Text>
                <Text style={styles.headerSubtitle}>{profile.email || 'No email on file'}</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        <Card style={styles.settingsCard}>
          <Card.Content>
            <Text variant="titleMedium" style={styles.sectionTitle}>
              Profile settings
            </Text>
            <TextInput
              mode="outlined"
              label="First name"
              value={profile.first_name}
              onChangeText={(value) => setProfile((prev) => ({ ...prev, first_name: value }))}
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Last name"
              value={profile.last_name}
              onChangeText={(value) => setProfile((prev) => ({ ...prev, last_name: value }))}
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Email"
              keyboardType="email-address"
              autoCapitalize="none"
              value={profile.email}
              onChangeText={(value) => setProfile((prev) => ({ ...prev, email: value }))}
              style={styles.input}
            />
            {successMessage ? <Text style={styles.successText}>{successMessage}</Text> : null}
            <Button
              mode="contained"
              onPress={handleSave}
              style={styles.saveButton}
              loading={saving}
              disabled={saving || loading}
            >
              Save changes
            </Button>
            {loading ? (
              <ActivityIndicator style={styles.loadingIndicator} size="small" color="#0F6B6E" />
            ) : null}
          </Card.Content>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
};

export default ProfileSettingsScreen;
