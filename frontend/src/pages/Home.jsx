import Hero     from "../components/Hero";
import Ticker   from "../components/Ticker";
import PinSection from "../components/PinSection";
import RaceRail from "../components/RaceRail";

export default function Home() {
  return (
    <>
      <Hero />
      <Ticker year={2026} />
      <PinSection />
      <RaceRail year={2026} />
    </>
  );
}
