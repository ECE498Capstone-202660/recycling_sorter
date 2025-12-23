import React, { useState } from "react";
import { View, TextInput, Text, TouchableOpacity } from "react-native";
import { useAuth } from "../../context/AuthContext";
import styles from './RegisterScreen.styles';

const RegisterScreen = ({ navigation }: any) => {
  const { register } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");

  const handleRegister = async () => {
    setError("");
    const normalize = (value: string) => {
      const trimmed = value.trim();
      return trimmed.length ? trimmed : null;
    };
    const success = await register(username, password, {
      first_name: normalize(firstName),
      last_name: normalize(lastName),
      email: normalize(email),
    });
    if (!success) setError("Registration failed");
  };

  return (
    <View style={styles.container}>
      <View style={styles.formCard}>
        <TextInput
          placeholder="Username"
          value={username}
          onChangeText={setUsername}
          style={styles.input}
          placeholderTextColor="#888"
        />
        <TextInput
          placeholder="First name"
          value={firstName}
          onChangeText={setFirstName}
          style={styles.input}
          placeholderTextColor="#888"
        />
        <TextInput
          placeholder="Last name"
          value={lastName}
          onChangeText={setLastName}
          style={styles.input}
          placeholderTextColor="#888"
        />
        <TextInput
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          style={styles.input}
          placeholderTextColor="#888"
          autoCapitalize="none"
          keyboardType="email-address"
        />
        <TextInput
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          style={styles.input}
          placeholderTextColor="#888"
        />
        {error ? <Text style={styles.error}>{error}</Text> : null}
        <TouchableOpacity style={styles.button} onPress={handleRegister}>
          <Text style={styles.buttonText}>REGISTER</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryButton} onPress={() => navigation.navigate("Login")}> 
          <Text style={styles.secondaryButtonText}>ALREADY HAVE AN ACCOUNT? LOGIN</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

export default RegisterScreen; 
