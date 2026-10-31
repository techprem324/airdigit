const localtunnel = require('localtunnel');
const fs = require('fs');
const https = require('https');

const PORT = 8000;
const SUBDOMAIN = 'airdigit-ai-' + Math.floor(1000 + Math.random() * 9000);

function getTunnelPassword() {
  return new Promise((resolve) => {
    https.get('https://loca.lt/mytunnelpassword', (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(data.trim()));
    }).on('error', () => resolve('152.58.163.84'));
  });
}

// Prevent Node process from exiting
const keepAlive = setInterval(() => {}, 1000 * 60 * 60);

async function start() {
  const password = await getTunnelPassword();

  async function connect() {
    try {
      console.log('Opening secure HTTPS tunnel on port ' + PORT + '...');
      const tunnel = await localtunnel({ port: PORT, subdomain: SUBDOMAIN });
      
      const info = `========================================================\n` +
                   `🚀 AIRDIGIT AI IS LIVE!\n` +
                   `🌐 Public Live URL: ${tunnel.url}\n` +
                   `🔑 Tunnel Password (if prompted): ${password}\n` +
                   `========================================================\n`;
      console.log(info);
      fs.writeFileSync('LIVE_URL.txt', info, 'utf-8');

      tunnel.on('close', () => {
        console.log('Tunnel connection closed. Reconnecting in 3s...');
        setTimeout(connect, 3000);
      });

      tunnel.on('error', (err) => {
        console.error('Tunnel error:', err.message);
        setTimeout(connect, 3000);
      });
    } catch (e) {
      console.error('Failed to establish tunnel:', e.message);
      setTimeout(connect, 4000);
    }
  }

  connect();
}

start();
