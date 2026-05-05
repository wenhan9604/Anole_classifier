import { useState, useEffect } from 'react';
import CountUp from 'react-countup';
import { motion } from 'framer-motion';

interface SpeciesStat {
  name: string;
  count: number;
  id: number;
}

interface ObserverStat {
  login: string;
  count: number;
  icon_url?: string | null;
}

interface DashboardStats {
  observations: number;
  species: number;
  contributors: number;
  activity: number[];
  top_observers: ObserverStat[];
  species_distribution: SpeciesStat[];
}

const SPECIES_CONFIG: Record<string, { scientificName: string; color: string; native: boolean }> = {
  'Green Anole': { scientificName: 'Anolis carolinensis', color: '#7fa14a', native: true },
  'Brown Anole': { scientificName: 'Anolis sagrei', color: '#8b6a3e', native: false },
  'Knight Anole': { scientificName: 'Anolis equestris', color: '#4a6b2a', native: false },
  'Bark Anole': { scientificName: 'Anolis distichus', color: '#a8774a', native: false },
  'Crested Anole': { scientificName: 'Anolis cristatellus', color: '#6b8fa5', native: false },
};

const getSpeciesConfig = (name: string) => {
  const match = Object.entries(SPECIES_CONFIG).find(([speciesName]) => (
    name === speciesName || name.toLowerCase().includes(speciesName.toLowerCase())
  ));
  return match?.[1] || { scientificName: 'Species record', color: '#6f7f52', native: false };
};

export default function AppMetrics() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const url = import.meta.env.VITE_API_URL 
          ? `${import.meta.env.VITE_API_URL}/api/metrics/dashboard`
          : '/api/metrics/dashboard';
          
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

  const totalSpeciesObservations = stats?.species_distribution.reduce((sum, species) => sum + species.count, 0) || 0;
  const maxActivity = Math.max(...(stats?.activity || [0]), 1);
  const recentActivity = stats?.activity.slice(-7).reduce((sum, count) => sum + count, 0) || 0;
  const hasStats = Boolean(stats && (stats.observations > 0 || stats.species_distribution.length > 0 || stats.top_observers.length > 0));

  const StatCard = ({ value, label, helper, index }: { value: number, label: string, helper: string, index: number }) => (
    <motion.div
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      style={{
        padding: "1rem",
        backgroundColor: "rgba(255, 255, 255, 0.86)",
        borderRadius: "14px",
        border: "1px solid rgba(57, 91, 42, 0.16)",
        boxShadow: "0 14px 30px rgba(42, 64, 35, 0.08)",
        textAlign: "left"
      }}
    >
      <div style={{ color: "#1f4f24", fontSize: "1.9rem", fontWeight: "800", lineHeight: 1 }}>
        <CountUp end={value} duration={2.5} separator="," />
      </div>
      <div style={{ color: "#254d2b", fontSize: "0.72rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.08em", marginTop: "0.45rem" }}>
        {label}
      </div>
      <div style={{ color: "#68755f", fontSize: "0.76rem", marginTop: "0.2rem" }}>{helper}</div>
    </motion.div>
  );

  if (error) {
    return (
      <section style={{ textAlign: "left" }}>
        <p style={{ margin: 0, color: "#50614a", fontSize: "0.9rem" }}>
          Community metrics are temporarily unavailable. Classification is still ready to use.
        </p>
      </section>
    );
  }

  if (!stats) {
    return (
      <section style={{ display: "grid", gap: "1rem" }}>
        {[0, 1, 2].map((item) => (
          <div key={item} style={{ height: 76, borderRadius: 14, background: "linear-gradient(90deg, rgba(255,255,255,0.55), rgba(255,255,255,0.9), rgba(255,255,255,0.55))", backgroundSize: "200% 100%", animation: "metricShimmer 1.4s infinite linear" }} />
        ))}
      </section>
    );
  }

  return (
    <div style={{
      width: "100%",
      display: "flex",
      flexDirection: "column",
      gap: "1.25rem"
    }}>
      <style>{`
        @keyframes metricShimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
      <motion.h4 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }}
        style={{ margin: 0, color: "#1b5e20", fontSize: "0.88rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.12em" }}
      >
        Community Impact
      </motion.h4>
      
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
        gap: "0.85rem"
      }}>
        <StatCard value={stats.observations} label="Observations" helper="iNaturalist records" index={0} />
        <StatCard value={stats.species} label="Species" helper="tracked in submissions" index={1} />
        <StatCard value={stats.contributors} label="Contributors" helper="active observers" index={2} />
      </div>

      {hasStats && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          style={{
            background: "rgba(255, 255, 255, 0.72)",
            border: "1px solid rgba(57, 91, 42, 0.14)",
            borderRadius: "16px",
            padding: "1rem",
            textAlign: "left",
            boxShadow: "0 12px 26px rgba(42, 64, 35, 0.06)"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "baseline", marginBottom: "0.7rem" }}>
            <div>
              <div style={{ color: "#1f4f24", fontWeight: 800, fontSize: "0.86rem" }}>Species breakdown</div>
              <div style={{ color: "#68755f", fontSize: "0.76rem" }}>Share of community observations</div>
            </div>
            <div style={{ color: "#68755f", fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.08em" }}>N={totalSpeciesObservations.toLocaleString()}</div>
          </div>

          {stats.species_distribution.length > 0 && totalSpeciesObservations > 0 ? (
            <>
              <div style={{ display: "flex", height: 34, overflow: "hidden", borderRadius: 999, border: "1px solid rgba(31, 79, 36, 0.22)", background: "#edf3e8" }}>
                {stats.species_distribution.map((species) => {
                  const config = getSpeciesConfig(species.name);
                  return (
                    <div
                      key={species.id || species.name}
                      title={`${species.name}: ${species.count}`}
                      style={{ flex: species.count, minWidth: 6, background: config.color }}
                    />
                  );
                })}
              </div>

              <div style={{ display: "grid", gap: "0.6rem", marginTop: "0.9rem" }}>
                {stats.species_distribution.slice(0, 5).map((species) => {
                  const config = getSpeciesConfig(species.name);
                  const percentage = Math.round((species.count / totalSpeciesObservations) * 100);
                  return (
                    <div key={species.id || species.name} style={{ display: "grid", gridTemplateColumns: "12px minmax(0, 1fr) auto", gap: "0.65rem", alignItems: "center" }}>
                      <span style={{ width: 10, height: 10, borderRadius: "50%", background: config.color, display: "inline-block" }} />
                      <div style={{ minWidth: 0 }}>
                        <div style={{ display: "flex", gap: "0.4rem", alignItems: "baseline", flexWrap: "wrap" }}>
                          <span style={{ color: "#213b24", fontWeight: 700, fontSize: "0.82rem" }}>{species.name}</span>
                          <span style={{ color: "#68755f", fontSize: "0.72rem", fontStyle: "italic" }}>{config.scientificName}</span>
                          {config.native && <span style={{ color: "#2e7d32", fontSize: "0.65rem", fontWeight: 800, textTransform: "uppercase" }}>native</span>}
                        </div>
                      </div>
                      <div style={{ color: "#365f37", fontSize: "0.78rem", fontWeight: 800 }}>{percentage}%</div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <p style={{ margin: 0, color: "#68755f", fontSize: "0.82rem" }}>Species data will appear once observations are available.</p>
          )}
        </motion.div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.9rem" }}>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45 }}
          style={{ background: "rgba(255, 255, 255, 0.68)", border: "1px solid rgba(57, 91, 42, 0.14)", borderRadius: "16px", padding: "1rem", textAlign: "left" }}
        >
          <div style={{ color: "#1f4f24", fontWeight: 800, fontSize: "0.82rem", marginBottom: "0.65rem" }}>Top contributors</div>
          {stats.top_observers.length > 0 ? (
            <div style={{ display: "grid", gap: "0.65rem" }}>
              {stats.top_observers.slice(0, 4).map((observer) => (
                <div key={observer.login} style={{ display: "grid", gridTemplateColumns: "28px minmax(0, 1fr) auto", gap: "0.55rem", alignItems: "center" }}>
                  <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#dfe9d8", overflow: "hidden", border: "1px solid rgba(57, 91, 42, 0.18)" }}>
                    {observer.icon_url ? <img src={observer.icon_url} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : null}
                  </div>
                  <span style={{ color: "#213b24", fontWeight: 700, fontSize: "0.78rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{observer.login}</span>
                  <span style={{ color: "#365f37", fontWeight: 800, fontSize: "0.76rem" }}>{observer.count}</span>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ margin: 0, color: "#68755f", fontSize: "0.8rem" }}>Contributor data will appear after community submissions.</p>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55 }}
          style={{ background: "rgba(255, 255, 255, 0.68)", border: "1px solid rgba(57, 91, 42, 0.14)", borderRadius: "16px", padding: "1rem", textAlign: "left" }}
        >
          <div style={{ color: "#1f4f24", fontWeight: 800, fontSize: "0.82rem" }}>Recent activity</div>
          <div style={{ color: "#1f4f24", fontSize: "1.8rem", fontWeight: 800, lineHeight: 1, marginTop: "0.6rem" }}>
            <CountUp end={recentActivity} duration={2.5} separator="," />
          </div>
          <div style={{ color: "#68755f", fontSize: "0.76rem", marginTop: "0.2rem" }}>observations in the last 7 days</div>
          {stats.activity.length > 0 && (
            <div style={{ display: "flex", alignItems: "end", gap: 4, height: 44, marginTop: "0.9rem" }}>
              {stats.activity.slice(-14).map((count, index) => (
                <div
                  key={`${count}-${index}`}
                  style={{
                    flex: 1,
                    minHeight: 4,
                    height: `${Math.max(8, (count / maxActivity) * 44)}px`,
                    borderRadius: "999px 999px 2px 2px",
                    background: index >= 7 ? "#4f7b3b" : "#afc3a0"
                  }}
                />
              ))}
            </div>
          )}
        </motion.div>
      </div>

      <motion.p 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.75 }}
        style={{ margin: 0, color: "#62705b", fontSize: "0.8rem", fontStyle: "italic" }}
      >
        Powered by community observations and iNaturalist submissions.
      </motion.p>
    </div>
  );
}
