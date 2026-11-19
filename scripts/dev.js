import { spawn } from 'child_process';

const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const nodeCmd = process.execPath;

function run(name, script) {
  const child = spawn(npmCmd, ['run', script], {
    stdio: 'inherit',
    env: process.env,
  });

  child.on('exit', (code) => {
    if (code && code !== 0) {
      console.error(`${name} exited with code ${code}`);
      shutdown(code);
    }
  });

  return child;
}

function runServer() {
  const child = spawn(nodeCmd, ['server.js'], {
    stdio: 'inherit',
    env: { ...process.env, SERVE_STATIC: 'false' },
  });

  child.on('exit', (code) => {
    if (code && code !== 0) {
      console.error(`server exited with code ${code}`);
      shutdown(code);
    }
  });

  return child;
}

const children = [runServer(), run('client', 'dev:client')];

function shutdown(code = 0) {
  for (const child of children) {
    if (!child.killed) {
      child.kill('SIGTERM');
    }
  }
  process.exit(code);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
