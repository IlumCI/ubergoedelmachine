//! Content hashing for the ledger chain and for policy versioning.
//!
//! Serialization order is the hash input, so [`Digest`] is only stable because
//! every hashed type derives `Serialize` with a fixed field order. Reordering
//! fields on a hashed struct changes historical digests; treat those layouts as
//! part of the on-disk format.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

/// A SHA-256 digest, rendered as lowercase hex when serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// The chain anchor. The first ledger entry links to this.
    pub const ZERO: Digest = Digest([0u8; 32]);

    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(bytes);
        let out = h.finalize();
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&out);
        Self(buf)
    }

    /// Hash of `prev || bytes`, the link step of the ledger chain.
    pub fn chain(prev: &Digest, bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(prev.0);
        h.update(bytes);
        let out = h.finalize();
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&out);
        Self(buf)
    }

    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }

    pub fn from_hex(s: &str) -> Option<Self> {
        let v = hex::decode(s).ok()?;
        let arr: [u8; 32] = v.try_into().ok()?;
        Some(Self(arr))
    }

    /// Short form for logs and CLI output.
    pub fn short(self) -> String {
        self.to_hex()[..12].to_string()
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Digest::from_hex(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("not a 32-byte hex digest: {s}")))
    }
}

/// Hash any serializable value through its JSON form.
///
/// Panics only if the value cannot be serialized at all, which for the types in
/// this crate is unreachable.
pub fn hash_json<T: Serialize>(value: &T) -> Digest {
    let bytes = serde_json::to_vec(value).expect("hashable types must serialize");
    Digest::of_bytes(&bytes)
}
