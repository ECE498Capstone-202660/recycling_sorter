import React, { useState } from "react";
import { View } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Button, Card, Text, TextInput } from "react-native-paper";
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
    const normalize = (value: string) => value.trim();
    const normalizedUsername = normalize(username);
    const normalizedPassword = normalize(password);
    const normalizedFirstName = normalize(firstName);
    const normalizedLastName = normalize(lastName);
    const normalizedEmail = normalize(email);

    if (
      !normalizedUsername ||
      !normalizedPassword ||
      !normalizedFirstName ||
      !normalizedLastName ||
      !normalizedEmail
    ) {
      setError("Please fill out all fields.");
      return;
    }
    const success = await register(username, password, {
      first_name: normalizedFirstName,
      last_name: normalizedLastName,
      email: normalizedEmail,
    });
    if (!success) setError("Registration failed");
  };

  return (
    <LinearGradient colors={["#E2F2F1", "#F6F7F4"]} style={styles.container}>
      <View style={styles.content}>
        <View style={styles.hero}>
          <Text variant="headlineMedium" style={styles.title}>
            Create your account
          </Text>
          <Text style={styles.subtitle}>
            Join the recycling journey in just a few steps.
          </Text>
        </View>

        <Card style={styles.formCard}>
          <Card.Content>
            <TextInput
              mode="outlined"
              placeholder="Username"
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
              style={styles.input}
              outlineStyle={styles.inputOutline}
              contentStyle={styles.inputContent}
              left={<TextInput.Icon icon="account" />}
            />
            <TextInput
              mode="outlined"
              placeholder="First name"
              value={firstName}
              onChangeText={setFirstName}
              style={styles.input}
              outlineStyle={styles.inputOutline}
              contentStyle={styles.inputContent}
              left={<TextInput.Icon icon="account-outline" />}
            />
            <TextInput
              mode="outlined"
              placeholder="Last name"
              value={lastName}
              onChangeText={setLastName}
              style={styles.input}
              outlineStyle={styles.inputOutline}
              contentStyle={styles.inputContent}
              left={<TextInput.Icon icon="account-outline" />}
            />
            <TextInput
              mode="outlined"
              placeholder="Email"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
              style={styles.input}
              outlineStyle={styles.inputOutline}
              contentStyle={styles.inputContent}
              left={<TextInput.Icon icon="email-outline" />}
            />
            <TextInput
              mode="outlined"
              placeholder="Password"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              style={styles.input}
              outlineStyle={styles.inputOutline}
              contentStyle={styles.inputContent}
              left={<TextInput.Icon icon="lock" />}
            />
            {error ? <Text style={styles.error}>{error}</Text> : null}
            <Button mode="contained" style={styles.button} onPress={handleRegister}>
              Register
            </Button>
            <Button
              mode="text"
              style={styles.secondaryButton}
              onPress={() => navigation.navigate("Login")}
            >
              Already have an account? Log in
            </Button>
          </Card.Content>
        </Card>
      </View>
    </LinearGradient>
  );
};

export default RegisterScreen; 
