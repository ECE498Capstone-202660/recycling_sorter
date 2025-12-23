import type { NavigatorScreenParams } from '@react-navigation/native';

export type DrawerParamList = {
  Home: undefined;
  History: undefined;
  Profile: undefined;
};

export type RootStackParamList = {
  Login: undefined;
  Register: undefined;
  Main: NavigatorScreenParams<DrawerParamList>;
  ProfileSettings: undefined;
};
