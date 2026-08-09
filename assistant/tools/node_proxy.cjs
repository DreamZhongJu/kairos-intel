// Node's built-in fetch does not honor HTTP(S)_PROXY by itself. The X fallback
// client uses fetch, so install an Undici dispatcher explicitly. This file
// contains no credentials: it consumes the container's existing proxy URL.
const { ProxyAgent, setGlobalDispatcher } = require("undici");

const proxyUrl = process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.ALL_PROXY;
if (proxyUrl) {
  setGlobalDispatcher(new ProxyAgent(proxyUrl));
}
