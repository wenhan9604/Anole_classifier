import { useState, useEffect } from 'react';
import CountUp from 'react-countup';
import { motion } from 'framer-motion';

interface Stats {
  observations: number;
  species: number;
  contributors: number;
}

export default function AppMetrics() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const url = import.meta.env.VITE_API_URL 
          ? `${import.meta.env.VITE_API_URL}/api/metrics/stats`
          : '/api/metrics/stats';
          
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch stats');
        const data = await response.json();
        setStats(data);
      } catch (err) {
        console.error("Error fetching app stats:", err);
        setError(true);
      }
    };

    fetchStats();
  }, []);

  if (error || !stats) return null;

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.15,
        type: 'spring' as const,
        stiffness: 100,
        damping: 12
      }
    })
  };

  const StatCard = ({ icon, value, label, index }: { icon: string, value: number, label: string, index: number }) => (
    <motion.div
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      style={{
        flex: "1 1 120px",
        padding: "1rem",
        backgroundColor: "rgba(255, 255, 255, 0.8)",
        backdropFilter: "blur(8px)",
        borderRadius: "16px",
        border: "1px solid rgba(76, 175, 80, 0.15)",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.05)",
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "0.25rem"
      }}
    >
      <span style={{ fontSize: "1.8rem", marginBottom: "0.25rem" }}>{icon}</span>
      <div style={{ color: "#1b5e20", fontSize: "1.8rem", fontWeight: "800", fontFamily: "'Inter', sans-serif" }}>
        <CountUp end={value} duration={2.5} separator="," />
      </div>
      <div style={{ color: "#2E7D32", fontSize: "0.75rem", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>
        {label}
      </div>
    </motion.div>
  );

  return (
    <div style={{
      marginTop: "0.5rem",
      padding: "0.5rem",
      display: "flex",
      flexDirection: "column",
      gap: "1.5rem"
    }}>
      <motion.h4 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }}
        style={{ margin: 0, color: "#1b5e20", fontSize: "1rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "1px" }}
      >
        🌍 Community Impact
      </motion.h4>
      
      <div style={{
        display: "flex",
        gap: "1rem",
        flexWrap: "wrap",
        justifyContent: "center"
      }}>
        <StatCard icon="📸" value={stats.observations} label="Observations" index={0} />
        <StatCard icon="🦎" value={stats.species} label="Species Found" index={1} />
        <StatCard icon="👥" value={stats.contributors} label="Contributors" index={2} />
      </div>

      <motion.p 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        style={{ margin: 0, color: "#666", fontSize: "0.85rem", fontStyle: "italic" }}
      >
        Proudly powered by the Lizard Lens community
      </motion.p>
    </div>
  );
}
