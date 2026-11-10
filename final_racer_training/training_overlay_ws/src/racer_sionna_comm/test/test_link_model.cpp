#include <racer_sionna_comm/link_model.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

TEST(LinkModel, StrongerSnrImprovesRateAndReliability) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200});
  EXPECT_GT(model.bitRate(23.0), model.bitRate(0.0));
  EXPECT_LT(model.packetErrorRate(23.0, 1200),
            model.packetErrorRate(0.0, 1200));
}

TEST(LinkModel, LargerMessagesHaveMoreErrorsAndSerializationDelay) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200});
  EXPECT_GT(model.packetErrorRate(8.0, 4800),
            model.packetErrorRate(8.0, 600));
  EXPECT_GT(model.serializationDelay(8.0, 4800),
            model.serializationDelay(8.0, 600));
}

TEST(LinkModel, ReliablePacketCapacityUsesAdaptiveMcsAndRetryCost) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200});

  // Within QPSK, a weak channel admits fewer reliable packets because the
  // expected retransmission airtime is higher.
  EXPECT_GT(model.reliablePacketCapacity(4.0, 1068, 3, 0.008),
            model.reliablePacketCapacity(-5.0, 1068, 3, 0.008));

  // A larger packet spans several PHY slots at QPSK, so adaptive 64QAM can
  // carry more packets in the same airtime.
  EXPECT_GT(model.reliablePacketCapacity(18.0, 10000, 3, 0.008),
            model.reliablePacketCapacity(4.0, 10000, 3, 0.008));
  EXPECT_GT(model.expectedReliablePacketAirtime(-5.0, 1068, 3),
            model.expectedReliablePacketAirtime(4.0, 1068, 3));
  EXPECT_THROW(model.reliablePacketCapacity(4.0, 1068, -1, 0.008),
               std::invalid_argument);
  EXPECT_THROW(model.reliablePacketCapacity(4.0, 1068, 3, -0.001),
               std::invalid_argument);
}

TEST(LinkModel, ReportsOriginalMtuPacketizationForStatistics) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200});
  EXPECT_EQ(model.transportBlockCount(0), 1U);
  EXPECT_EQ(model.transportBlockCount(1), 1U);
  EXPECT_EQ(model.transportBlockCount(1200), 1U);
  EXPECT_EQ(model.transportBlockCount(1201), 2U);
  EXPECT_EQ(model.transportBlockCount(4800), 4U);
}

TEST(LinkModel, UsesOnlyRequestedAdaptiveModulations) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200});
  EXPECT_EQ(model.selectMcs(-5.0).name, "QPSK");
  EXPECT_EQ(model.selectMcs(7.0).name, "16QAM");
  EXPECT_EQ(model.selectMcs(14.0).name, "64QAM");
  EXPECT_EQ(model.selectMcs(21.0).name, "256QAM");
  EXPECT_NEAR(model.transportBlockErrorRate(14.0), 0.10, 1.0e-12);
  EXPECT_NEAR(model.slotDuration(), 0.000125, 1.0e-12);
}

TEST(LinkModel, CanFixNrTableOneMcs14) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200, 14});
  const auto &low_snr = model.selectMcs(-20.0);
  const auto &high_snr = model.selectMcs(40.0);
  EXPECT_EQ(low_snr.name, "16QAM");
  EXPECT_EQ(high_snr.name, "16QAM");
  EXPECT_EQ(low_snr.modulation_order, 4);
  EXPECT_NEAR(low_snr.code_rate, 616.0 / 1024.0, 1.0e-12);
  EXPECT_NEAR(model.transportBlockErrorRate(9.0), 0.10, 1.0e-12);
  EXPECT_DOUBLE_EQ(model.bitRate(-20.0), model.bitRate(40.0));
}

TEST(LinkModel, CanFixNrTableOneMcs20) {
  racer_sionna_comm::LinkModel model(
      {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200, 20});
  const auto &low_snr = model.selectMcs(-20.0);
  const auto &high_snr = model.selectMcs(40.0);
  EXPECT_EQ(low_snr.name, "64QAM");
  EXPECT_EQ(high_snr.name, "64QAM");
  EXPECT_EQ(low_snr.modulation_order, 6);
  EXPECT_NEAR(low_snr.code_rate, 567.0 / 1024.0, 1.0e-12);
  EXPECT_NEAR(model.transportBlockErrorRate(13.0), 0.10, 1.0e-12);
  EXPECT_DOUBLE_EQ(model.bitRate(-20.0), model.bitRate(40.0));
  EXPECT_NEAR(model.bitRate(13.0, 22), model.bitRate(13.0) / 3.0,
              1.0e-9);
  EXPECT_NEAR(model.bitsPerSlot(13.0, 66),
              model.bitRate(13.0) * model.slotDuration(), 1.0e-9);
  EXPECT_THROW(model.bitRate(13.0, 0), std::invalid_argument);
  EXPECT_THROW(model.bitRate(13.0, 67), std::invalid_argument);
}

TEST(LinkModel, RejectsInvalidConfiguration) {
  EXPECT_THROW(
      racer_sionna_comm::LinkModel(
          {0.0, 120.0e3, 66, 0.82, 0.10, 1.35, 1200}),
      std::invalid_argument);
  EXPECT_THROW(
      racer_sionna_comm::LinkModel(
          {100.0e6, 120.0e3, 66, 0.82, 0.10, 1.35, 1200, 15}),
      std::invalid_argument);
}
