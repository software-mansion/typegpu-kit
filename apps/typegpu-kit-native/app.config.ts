import type { ConfigContext } from 'expo/config';

export default ({ config }: ConfigContext) => ({
  ...config,
  ios: { ...config.ios, appleTeamId: process.env.APPLE_TEAM_ID },
});
