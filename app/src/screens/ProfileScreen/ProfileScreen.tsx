import React, { useEffect, useMemo, useState } from 'react';
import { View, ScrollView, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { MaterialIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { ActivityIndicator, Button, Card, Surface, Text } from 'react-native-paper';
import styles from './ProfileScreen.styles';
import { useAuth } from '../../context/AuthContext';
import { getUserMe } from '../../services/api';
import { RootStackParamList } from '../../types/navigation';

const LogoutIcon = ({ size, color }: { size: number; color: string }) => (
  <MaterialIcons name="logout" size={size} color={color} />
);

const ProfileScreen = () => {
  const { logout, token } = useAuth();
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [profile, setProfile] = useState({
    username: '',
    email: '',
    first_name: '',
    last_name: '',
  });
  const [loading, setLoading] = useState(true);

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

  const menuItems = [
    {
      id: '1',
      title: 'Profile Settings',
      icon: 'person',
      onPress: () => navigation.navigate('ProfileSettings'),
    },
    {
      id: '2',
      title: 'Notifications',
      icon: 'notifications',
      onPress: () => {},
    },
    {
      id: '3',
      title: 'Privacy Policy',
      icon: 'security',
      onPress: () => {},
    },
    {
      id: '4',
      title: 'Help & Support',
      icon: 'help',
      onPress: () => {},
    },
    {
      id: '5',
      title: 'About',
      icon: 'info',
      onPress: () => {},
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <LinearGradient colors={['#E2F2F1', '#F6F7F4']} style={styles.hero}>
          <View style={styles.heroContent}>
            <View style={styles.profileImage}>
              <MaterialIcons name="person" size={44} color="#0F6B6E" />
            </View>
            <Text variant="titleLarge" style={styles.name}>
              {displayName}
            </Text>
            <Text style={styles.email}>{profile.email || 'No email on file'}</Text>
          </View>
        </LinearGradient>

        <View style={styles.statsRow}>
          <Surface style={styles.statCard} elevation={0}>
            <Text style={styles.statValue}>156</Text>
            <Text style={styles.statLabel}>Points</Text>
          </Surface>
          <Surface style={styles.statCard} elevation={0}>
            <Text style={styles.statValue}>23</Text>
            <Text style={styles.statLabel}>Items recycled</Text>
          </Surface>
          <Surface style={styles.statCard} elevation={0}>
            <Text style={styles.statValue}>$12.50</Text>
            <Text style={styles.statLabel}>Rebate earned</Text>
          </Surface>
        </View>

        <Card style={styles.menuCard}>
          <Card.Content>
            {menuItems.map((item, index) => (
              <TouchableOpacity
                key={item.id}
                style={[
                  styles.menuItem,
                  index < menuItems.length - 1 ? styles.menuDivider : null,
                ]}
                onPress={item.onPress}
              >
                <View style={styles.menuItemLeft}>
                  <MaterialIcons name={item.icon as any} size={22} color="#0F6B6E" />
                  <Text style={styles.menuItemText}>{item.title}</Text>
                </View>
                <MaterialIcons name="chevron-right" size={22} color="#8A9899" />
              </TouchableOpacity>
            ))}
          </Card.Content>
        </Card>

        <Button
          mode="contained"
          onPress={logout}
          style={styles.logoutButton}
          contentStyle={styles.logoutContent}
          icon={LogoutIcon}
        >
          Log out
        </Button>
        {loading ? (
          <ActivityIndicator style={styles.loadingIndicator} size="small" color="#0F6B6E" />
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
};

export default ProfileScreen; 
