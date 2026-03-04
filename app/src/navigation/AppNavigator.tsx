import React from 'react';
import { LinkingOptions, NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {
  createDrawerNavigator,
  DrawerContentScrollView,
  DrawerItemList,
} from '@react-navigation/drawer';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { DrawerParamList, RootStackParamList } from '../types/navigation';
import LoginScreen from "../screens/LoginScreen/LoginScreen";
import RegisterScreen from "../screens/RegisterScreen/RegisterScreen";
import { useAuth } from "../context/AuthContext";

// Import screens
import HomeScreen from '../screens/HomeScreen/HomeScreen';
import HistoryScreen from '../screens/HistoryScreen/HistoryScreen';
import ProfileScreen from '../screens/ProfileScreen/ProfileScreen';
import ProfileSettingsScreen from '../screens/ProfileSettingsScreen/ProfileSettingsScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();
const Drawer = createDrawerNavigator<DrawerParamList>();

const linking: LinkingOptions<RootStackParamList> = {
  prefixes: ['http://localhost:8081', 'http://127.0.0.1:8081', 'http://192.168.0.185:8081'],
  config: {
    screens: {
      Login: 'login',
      Register: 'register',
      Main: {
        screens: {
          Home: '',
          History: 'history',
          Profile: 'profile',
        },
      },
      ProfileSettings: 'profile/settings',
    },
  },
};

const CustomDrawerContent = (props: any) => {
  const { logout } = useAuth();

  return (
    <View style={styles.drawerContainer}>
      <DrawerContentScrollView {...props} contentContainerStyle={styles.drawerContent}>
        <DrawerItemList {...props} />
      </DrawerContentScrollView>
      <TouchableOpacity style={styles.logoutButton} onPress={logout}>
        <Text style={styles.logoutText}>Log out</Text>
      </TouchableOpacity>
    </View>
  );
};

const MainDrawer = () => {
  return (
      <Drawer.Navigator
        initialRouteName="Home"
        screenOptions={{
        headerShown: true,
        headerTintColor: '#0B2F33',
        headerTitleStyle: { fontWeight: '600' },
        drawerActiveTintColor: '#0F6B6E',
      }}
        drawerContent={(props) => <CustomDrawerContent {...props} />}
    >
      <Drawer.Screen
        name="Home"
        component={HomeScreen}
        options={{ title: "Recycling Station" }}
      />
      <Drawer.Screen name="History" component={HistoryScreen} />
      <Drawer.Screen name="Profile" component={ProfileScreen} />
    </Drawer.Navigator>
  );
};

const AppNavigator = () => {
  const { token, isBootstrapped } = useAuth();

  if (!isBootstrapped) {
    return null;
  }

  return (
    <NavigationContainer linking={linking}>
      <Stack.Navigator>
        {token ? (
          <>
            <Stack.Screen name="Main" component={MainDrawer} options={{ headerShown: false }} />
            <Stack.Screen
              name="ProfileSettings"
              component={ProfileSettingsScreen}
              options={{ title: "Profile Settings" }}
            />
          </>
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Register" component={RegisterScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  drawerContainer: {
    flex: 1,
  },
  drawerContent: {
    paddingTop: 8,
  },
  logoutButton: {
    marginHorizontal: 16,
    marginBottom: 20,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: '#0F6B6E',
    alignItems: 'center',
  },
  logoutText: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
});

export default AppNavigator; 
