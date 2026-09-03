import test from "node:test";
import assert from "node:assert/strict";

import {
  CHAT_RESPONSE_TIMEOUT_STORAGE_KEY,
  DEFAULT_CHAT_RESPONSE_TIMEOUT_SECONDS,
  readStoredChatResponseTimeout,
  writeStoredChatResponseTimeout,
} from "../context/app-shell-storage";

const VERSION_KEY = "deeptutor.chatResponseTimeout.version";

function withLocalStorage(
  entries: Record<string, string>,
  run: (store: Map<string, string>) => void,
) {
  const store = new Map(Object.entries(entries));
  const original = (globalThis as { window?: unknown }).window;
  (globalThis as { window?: unknown }).window = {
    localStorage: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => void store.set(key, value),
    },
  };
  try {
    run(store);
  } finally {
    (globalThis as { window?: unknown }).window = original;
  }
}

test("chat timeout defaults beyond the five-minute provider deadline", () => {
  withLocalStorage({}, (store) => {
    assert.equal(DEFAULT_CHAT_RESPONSE_TIMEOUT_SECONDS, 360);
    assert.equal(readStoredChatResponseTimeout(), 360);
    assert.equal(store.get(CHAT_RESPONSE_TIMEOUT_STORAGE_KEY), "360");
    assert.equal(store.get(VERSION_KEY), "2");
  });
});

test("chat timeout migrates the old 180-second default once", () => {
  withLocalStorage({ [CHAT_RESPONSE_TIMEOUT_STORAGE_KEY]: "180" }, (store) => {
    assert.equal(readStoredChatResponseTimeout(), 360);
    assert.equal(store.get(CHAT_RESPONSE_TIMEOUT_STORAGE_KEY), "360");
    assert.equal(store.get(VERSION_KEY), "2");
  });
});

test("chat timeout preserves a deliberate legacy custom value", () => {
  withLocalStorage({ [CHAT_RESPONSE_TIMEOUT_STORAGE_KEY]: "900" }, (store) => {
    assert.equal(readStoredChatResponseTimeout(), 900);
    assert.equal(store.get(CHAT_RESPONSE_TIMEOUT_STORAGE_KEY), "900");
    assert.equal(store.get(VERSION_KEY), "2");
  });
});

test("chat timeout writer records the current storage version", () => {
  withLocalStorage({}, (store) => {
    writeStoredChatResponseTimeout(180);
    assert.equal(readStoredChatResponseTimeout(), 180);
    assert.equal(store.get(CHAT_RESPONSE_TIMEOUT_STORAGE_KEY), "180");
    assert.equal(store.get(VERSION_KEY), "2");
  });
});
