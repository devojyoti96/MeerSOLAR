#include "progressbar.h"
#include <iomanip>   // for std::setw and std::setfill
#include <ctime>     // for std::tm and std::gmtime
#include <sstream>   // for std::ostringstream
#include <casacore/ms/MeasurementSets/MeasurementSet.h>

#include <casacore/measures/TableMeasures/ArrayMeasColumn.h>
#include <casacore/measures/TableMeasures/ScalarMeasColumn.h>

#include <casacore/measures/Measures/MBaseline.h>
#include <casacore/measures/Measures/MCBaseline.h>
#include <casacore/measures/Measures/MCPosition.h>
#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MCuvw.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/measures/Measures/MCEpoch.h>
#include <casacore/measures/Measures/MPosition.h>
#include <casacore/measures/Measures/Muvw.h>
#include <casacore/measures/Measures/MeasConvert.h>
#include <casacore/measures/Measures/MeasTable.h>
#include <casacore/measures/Measures/SolarPos.h>

#include <casacore/tables/Tables/TableRecord.h>

#include <aocommon/banddata.h>
#include <aocommon/imagecoordinates.h>
#include <aocommon/multibanddata.h>
#include <aocommon/radeccoord.h>

#include <iostream>
#include <memory>

using casacore::ArrayColumn;
using casacore::MBaseline;
using casacore::MDirection;
using casacore::MeasFrame;
using casacore::MeasTable;
using casacore::MEpoch;
using casacore::MPosition;

using casacore::MSAntenna;
using casacore::MSAntennaEnums;
using casacore::MSField;
using casacore::MSFieldEnums;
using casacore::MSMainEnums;
using casacore::MSObservation;
using casacore::Muvw;
using casacore::MVBaseline;
using casacore::MVDirection;
using casacore::MVEpoch;
using casacore::MVPosition;
using casacore::MVuvw;
using casacore::ScalarColumn;

using aocommon::RaDecCoord;

std::vector<MPosition> antennas;

// typedef long int integer;
// typedef double doublereal;
typedef int integer;
typedef double doublereal;

extern "C" {
extern void dgesvd_(const char* jobu, const char* jobvt, const integer* M,
                    const integer* N, doublereal* A, const integer* lda,
                    doublereal* S, doublereal* U, const integer* ldu,
                    doublereal* VT, const integer* ldvt, doublereal* work,
                    const integer* lwork, const integer* info);
}

std::string dirToString(const MDirection& direction) {
  double ra = direction.getAngle().getValue()[0];
  double dec = direction.getAngle().getValue()[1];
  return RaDecCoord::RAToString(ra) + " " + RaDecCoord::DecToString(dec);
}

std::string posToString(const MPosition& position) {
  double lon = position.getAngle().getValue()[0] * 180.0 / M_PI;
  double lat = position.getAngle().getValue()[1] * 180.0 / M_PI;
  std::stringstream str;
  str << "Lon=" << lon << ", lat=" << lat;
  return str.str();
}

double length(const Muvw& uvw) {
  const casacore::Vector<double> v = uvw.getValue().getVector();
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double length(const casacore::Vector<double>& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Muvw calculateUVW(const MPosition& antennaPos, const MPosition& refPos,
                  const MEpoch& time, const MDirection& direction) {
  const casacore::Vector<double> posVec = antennaPos.getValue().getVector();
  const casacore::Vector<double> refVec = refPos.getValue().getVector();
  MVPosition relativePos(posVec[0] - refVec[0], posVec[1] - refVec[1],
                         posVec[2] - refVec[2]);
  MeasFrame frame(time, refPos, direction);
  MBaseline baseline(MVBaseline(relativePos),
                     MBaseline::Ref(MBaseline::ITRF, frame));
  MBaseline j2000Baseline = MBaseline::Convert(baseline, MBaseline::J2000)();
  MVuvw uvw(j2000Baseline.getValue(), direction.getValue());
  return Muvw(uvw, Muvw::J2000);
}

void RotateVisibilities(const aocommon::BandData& bandData, double shiftFactor,
                        unsigned polarizationCount,
                        casacore::Array<casacore::Complex>::contiter dataIter) {
  for (unsigned ch = 0; ch != bandData.ChannelCount(); ++ch) {
    const double wShiftRad = shiftFactor / bandData.ChannelWavelength(ch);
    double rotSin = std::sin(wShiftRad), rotCos = std::cos(wShiftRad);
    for (unsigned p = 0; p != polarizationCount; ++p) {
      casacore::Complex v = *dataIter;
      *dataIter = casacore::Complex(v.real() * rotCos - v.imag() * rotSin,
                                    v.real() * rotSin + v.imag() * rotCos);
      ++dataIter;
    }
  }
}

casacore::MPosition ArrayCentroid(casacore::MeasurementSet& set) {
  casacore::MSAntenna aTable = set.antenna();
  if (aTable.nrow() == 0) throw std::runtime_error("No antennae in set");
  casacore::MPosition::ScalarColumn antPosColumn(
      aTable, aTable.columnName(casacore::MSAntennaEnums::POSITION));
  double x = 0.0, y = 0.0, z = 0.0;
  for (size_t row = 0; row != aTable.nrow(); ++row) {
    casacore::MPosition antPos = antPosColumn(row);
    casacore::Vector<casacore::Double> vec = antPos.getValue().getVector();
    x += vec[0];
    y += vec[1];
    z += vec[2];
  }
  casacore::MPosition arrayPos = antPosColumn(0);
  double count = aTable.nrow();
  arrayPos.set(casacore::MVPosition(x / count, y / count, z / count),
               arrayPos.getRef());
  return arrayPos;
}

casacore::MPosition ArrayPosition(casacore::MeasurementSet& set,
                                  bool fallBackToCentroid = false) {
  static bool hasArrayPos = false;
  static MPosition arrayPos;
  if (!hasArrayPos) {
    MSObservation obsTable(set.observation());
    ScalarColumn<casacore::String> telescopeNameColumn(
        obsTable, obsTable.columnName(casacore::MSObservation::TELESCOPE_NAME));
    bool hasTelescopeName = telescopeNameColumn.nrow() != 0;
    bool hasObservatoryInfo = false;
    if (hasTelescopeName) {
      std::string telescopeName = telescopeNameColumn(0);
      hasObservatoryInfo = MeasTable::Observatory(arrayPos, telescopeName);
    }
    if (!hasTelescopeName || !hasObservatoryInfo) {
      if (!hasTelescopeName)
        std::cout << "WARNING: This set did not specify an observatory name.\n";
      else
        std::cout << "WARNING: Set specifies '" << telescopeNameColumn(0)
                  << "' as observatory name, but the array position of this "
                     "telescope is unknown.\n";

      // If not found, use the position of the first antenna or array centroid
      // if requested.
      if (fallBackToCentroid) {
        arrayPos = ArrayCentroid(set);
        std::cout << "Using antennae centroid as telescope position: "
                  << posToString(arrayPos) << '\n';
      } else {
        arrayPos = antennas[0];
        std::cout << "Using first antenna as telescope position: "
                  << posToString(arrayPos) << '\n';
      }
    } else {
      arrayPos = MPosition::Convert(arrayPos, MPosition::ITRF)();
      // std::cout << "Found '" << telescopeNameColumn(0) << "' array centre: "
      // << posToString(arrayPos) << " (first antenna is at " <<
      // posToString(antennas[0]) << ").\n";
    }
    hasArrayPos = true;
  }
  return arrayPos;
}

casacore::MDirection SolarDirection(casacore::MeasurementSet& set, size_t rowIndex) {
  using namespace casacore;

  // 1. Get time in MJD
  MEpoch::ScalarColumn timeColumn(set, set.columnName(MSMainEnums::TIME));
  MEpoch time = timeColumn(rowIndex);
  Double mjd = time.getValue().get();

  // 2. Use SolarPos operator()(epoch) to get MVPosition of Sun
  SolarPos solar;
  const MVPosition& sunVec = solar(mjd);

  // 3. Convert MVPosition to spherical coords (RA/Dec)
  //    Note: atan2 and asin give us equatorial coordinates in radians
  Vector<Double> cart = sunVec.getValue(); // x, y, z
  Double x = cart[0], y = cart[1], z = cart[2];
  Double r = std::sqrt(x * x + y * y + z * z);
  Double ra = std::atan2(y, x);
  if (ra < 0) ra += 2 * C::pi;
  Double dec = std::asin(z / r);

  // 4. Wrap in MDirection (J2000)
  return MDirection(MVDirection(ra, dec), MDirection::Ref(MDirection::J2000));
}

MDirection ZenithDirection(casacore::MeasurementSet& set, size_t rowIndex) {
  casacore::MPosition arrayPos = ArrayPosition(set);
  casacore::MEpoch::ScalarColumn timeColumn(
      set, set.columnName(casacore::MSMainEnums::TIME));
  casacore::MEpoch time = timeColumn(rowIndex);
  casacore::MeasFrame frame(arrayPos, time);
  const casacore::MDirection::Ref azelRef(casacore::MDirection::AZEL, frame);
  const casacore::MDirection::Ref j2000Ref(casacore::MDirection::J2000, frame);
  casacore::MDirection zenithAzEl(casacore::MVDirection(0.0, 0.0, 1.0),
                                  azelRef);
  return casacore::MDirection::Convert(zenithAzEl, j2000Ref)();
}

MDirection ZenithDirection(casacore::MeasurementSet& set) {
  return ZenithDirection(set, set.nrow() / 2);
}

MDirection ZenithDirectionStart(casacore::MeasurementSet& set) {
  return ZenithDirection(set, 0);
}

MDirection ZenithDirectionEnd(casacore::MeasurementSet& set) {
  return ZenithDirection(set, set.nrow() - 1);
}

void getShift(MSField& fieldTable, double& dl, double& dm) {
  if (fieldTable.keywordSet().isDefined("WSCLEAN_DL"))
    dl =
        fieldTable.keywordSet().asDouble(casacore::RecordFieldId("WSCLEAN_DL"));
  else
    dl = 0.0;
  if (fieldTable.keywordSet().isDefined("WSCLEAN_DM"))
    dm =
        fieldTable.keywordSet().asDouble(casacore::RecordFieldId("WSCLEAN_DM"));
  else
    dm = 0.0;
}

void processField(casacore::MeasurementSet& set, const std::string& dataColumn,
                  int fieldIndex, MSField& fieldTable,
                  const MDirection& newDirection, bool onlyUVW, bool shiftback,
                  double newDl, double newDm, bool flipUVWSign, bool force) {
  aocommon::MultiBandData bandData(set.spectralWindow(), set.dataDescription());
  ScalarColumn<casacore::String> nameCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::NAME));
  MDirection::ArrayColumn phaseDirCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::PHASE_DIR));
  MEpoch::ScalarColumn timeCol(set, set.columnName(MSMainEnums::TIME));

  ScalarColumn<int> antenna1Col(set, set.columnName(MSMainEnums::ANTENNA1)),
      antenna2Col(set, set.columnName(MSMainEnums::ANTENNA2)),
      fieldIdCol(set, set.columnName(MSMainEnums::FIELD_ID)),
      dataDescIdCol(set, set.columnName(MSMainEnums::DATA_DESC_ID));
  Muvw::ScalarColumn uvwCol(set, set.columnName(MSMainEnums::UVW));
  ArrayColumn<double> uvwOutCol(set, set.columnName(MSMainEnums::UVW));

  const bool hasCorrData = dataColumn.empty() &&
                           set.isColumn(casacore::MSMainEnums::CORRECTED_DATA),
             hasModelData = dataColumn.empty() &&
                            set.isColumn(casacore::MSMainEnums::MODEL_DATA);
  std::unique_ptr<ArrayColumn<casacore::Complex>> dataCol, correctedDataCol,
      modelDataCol;
  if (!onlyUVW) {
    if (dataColumn.empty()) {
      dataCol.reset(new ArrayColumn<casacore::Complex>(
          set, set.columnName(MSMainEnums::DATA)));
      if (hasCorrData) {
        correctedDataCol.reset(new ArrayColumn<casacore::Complex>(
            set, set.columnName(MSMainEnums::CORRECTED_DATA)));
      }
      if (hasModelData) {
        modelDataCol.reset(new ArrayColumn<casacore::Complex>(
            set, set.columnName(MSMainEnums::MODEL_DATA)));
      }
    } else {
      dataCol.reset(new ArrayColumn<casacore::Complex>(set, dataColumn));
    }
  }

  MPosition arrayPos = ArrayPosition(set);
  casacore::Vector<MDirection> phaseDirVector = phaseDirCol(fieldIndex);
  MDirection phaseDirection = phaseDirVector[0];
  double oldRA = phaseDirection.getAngle().getValue()[0];
  double oldDec = phaseDirection.getAngle().getValue()[1];
  double newRA = newDirection.getAngle().getValue()[0];
  double newDec = newDirection.getAngle().getValue()[1];
  if (shiftback)
    aocommon::ImageCoordinates::RaDecToLM(oldRA, oldDec, newRA, newDec, newDl,
                                          newDm);
  double oldDl, oldDm;
  getShift(fieldTable, oldDl, oldDm);
  std::cout << "Processing field \"" << nameCol(fieldIndex)
            << "\": " << dirToString(phaseDirection) << " -> "
            << dirToString(newDirection) << " ("
            << aocommon::ImageCoordinates::AngularDistance(oldRA, oldDec, newRA,
                                                           newDec) *
                   (180.0 / M_PI)
            << " deg)\n";
  if (oldDl != 0.0 || oldDm != 0.0 || newDl != 0.0 || newDm != 0.0) {
    std::cout << "Denormal shifting: " << oldDl << ',' << oldDm << " -> "
              << newDl << ',' << newDm;
  }
  std::cout << '\n';
  bool isSameDirection =
      (dirToString(phaseDirection) == dirToString(newDirection)) &&
      oldDl == newDl && oldDm == newDm;
  if (isSameDirection && !force) {
    std::cout << "Phase centre did not change: skipping field.\n";
  } else {
    if (isSameDirection)
      std::cout << "Phase centre not changed, but forcing update.\n";

    MDirection refDirection =
        MDirection::Convert(newDirection, MDirection::Ref(MDirection::J2000))();
    casacore::IPosition dataShape;
    unsigned polarizationCount = 0;
    std::unique_ptr<casacore::Array<casacore::Complex>> dataArray;
    if (!onlyUVW) {
      dataShape = dataCol->shape(0);
      polarizationCount = dataShape[0];
      dataArray.reset(new casacore::Array<casacore::Complex>(dataShape));
    }

    std::unique_ptr<ProgressBar> progressBar;

    std::vector<Muvw> uvws(antennas.size());
    MEpoch time(MVEpoch(-1.0));
    for (unsigned row = 0; row != set.nrow(); ++row) {
      if (fieldIdCol(row) == fieldIndex) {
        // Read values from set
        const int antenna1 = antenna1Col(row), antenna2 = antenna2Col(row),
                  dataDescId = dataDescIdCol(row);
        const Muvw oldUVW = uvwCol(row);

        MEpoch rowTime = timeCol(row);
        if (rowTime.getValue() != time.getValue()) {
          time = rowTime;
          for (size_t a = 0; a != antennas.size(); ++a)
            uvws[a] = calculateUVW(antennas[a], arrayPos, time, refDirection);
        }

        // Calculate the new UVW
        MVuvw newUVW = uvws[antenna1].getValue() - uvws[antenna2].getValue();
        if (flipUVWSign) newUVW = -newUVW;

        // If one of the first results, output values for analyzing them.
        if (row < 5) {
          std::cout << "Old " << oldUVW << " (" << length(oldUVW) << ")\n";
          std::cout << "New " << newUVW << " (" << length(newUVW) << ")\n\n";
        } else {
          if (progressBar == nullptr)
            progressBar.reset(new ProgressBar("Changing phase centre"));
          progressBar->SetProgress(row, set.nrow());
        }

        if (!onlyUVW) {
          // Read the visibilities and phase-rotate them
          double shiftFactor =
              -2.0 * M_PI *
              (newUVW.getVector()[2] - oldUVW.getValue().getVector()[2]);

          double dnu = newUVW.getVector()[0], dnv = newUVW.getVector()[1];
          shiftFactor += -2.0 * M_PI * (dnu * newDl + dnv * newDm);
          double dou = oldUVW.getValue().getVector()[0],
                 dov = oldUVW.getValue().getVector()[1];
          shiftFactor -= -2.0 * M_PI * (dou * oldDl + dov * oldDm);

          const aocommon::BandData& thisBand = bandData[dataDescId];
          dataCol->get(row, *dataArray);
          RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                             dataArray->cbegin());
          dataCol->put(row, *dataArray);

          if (hasCorrData) {
            correctedDataCol->get(row, *dataArray);
            RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                               dataArray->cbegin());
            correctedDataCol->put(row, *dataArray);
          }

          if (hasModelData) {
            modelDataCol->get(row, *dataArray);
            RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                               dataArray->cbegin());
            modelDataCol->put(row, *dataArray);
          }
        }

        // Store uvws
        uvwOutCol.put(row, newUVW.getVector());
      }
    }
    progressBar.reset();

    phaseDirVector[0] = newDirection;
    phaseDirCol.put(fieldIndex, phaseDirVector);

    if (newDl == 0.0 && newDm == 0.0) {
      if (fieldTable.keywordSet().isDefined("WSCLEAN_DL")) {
        fieldTable.rwKeywordSet().removeField(
            casacore::RecordFieldId("WSCLEAN_DL"));
        std::cout << "Removing WSCLEAN_DL keyword.\n";
      }
      if (fieldTable.keywordSet().isDefined("WSCLEAN_DM")) {
        fieldTable.rwKeywordSet().removeField(
            casacore::RecordFieldId("WSCLEAN_DM"));
        std::cout << "Removing WSCLEAN_DM keyword.\n";
      }
    } else {
      fieldTable.rwKeywordSet().define(casacore::RecordFieldId("WSCLEAN_DL"),
                                       newDl);
      fieldTable.rwKeywordSet().define(casacore::RecordFieldId("WSCLEAN_DM"),
                                       newDm);
    }
  }
}

void showChanges(casacore::MeasurementSet& set, int fieldIndex,
                 MSField& fieldTable, const MDirection& newDirection,
                 bool flipUVWSign) {
  ScalarColumn<casacore::String> nameCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::NAME));
  MDirection::ArrayColumn phaseDirCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::PHASE_DIR));
  MEpoch::ScalarColumn timeCol(set, set.columnName(MSMainEnums::TIME));

  ScalarColumn<int> antenna1Col(set, set.columnName(MSMainEnums::ANTENNA1)),
      antenna2Col(set, set.columnName(MSMainEnums::ANTENNA2)),
      fieldIdCol(set, set.columnName(MSMainEnums::FIELD_ID));
  Muvw::ScalarColumn uvwCol(set, set.columnName(MSMainEnums::UVW));

  MPosition arrayPos = ArrayPosition(set);
  casacore::Vector<MDirection> phaseDirVector = phaseDirCol(fieldIndex);
  MDirection phaseDirection = phaseDirVector[0];
  double oldRA = phaseDirection.getAngle().getValue()[0];
  double oldDec = phaseDirection.getAngle().getValue()[1];
  double newRA = newDirection.getAngle().getValue()[0];
  double newDec = newDirection.getAngle().getValue()[1];
  std::cout << "Showing UVWs for \"" << nameCol(fieldIndex)
            << "\": " << dirToString(phaseDirection) << " -> "
            << dirToString(newDirection) << " ("
            << aocommon::ImageCoordinates::AngularDistance(oldRA, oldDec, newRA,
                                                           newDec) *
                   (180.0 / M_PI)
            << " deg)\n";

  MDirection refDirection =
      MDirection::Convert(newDirection, MDirection::Ref(MDirection::J2000))();
  std::vector<Muvw> uvws(antennas.size());
  MEpoch time(MVEpoch(-1.0));
  for (size_t row = 0; row != std::min<size_t>(set.nrow(), 50u); ++row) {
    if (fieldIdCol(row) == fieldIndex) {
      // Read values from set
      const int antenna1 = antenna1Col(row), antenna2 = antenna2Col(row);
      const Muvw oldUVW = uvwCol(row);

      MEpoch rowTime = timeCol(row);
      if (rowTime.getValue() != time.getValue()) {
        time = rowTime;
        for (size_t a = 0; a != antennas.size(); ++a)
          uvws[a] = calculateUVW(antennas[a], arrayPos, time, refDirection);
      }

      // Calculate the new UVW
      MVuvw newUVW = uvws[antenna1].getValue() - uvws[antenna2].getValue();
      if (flipUVWSign) newUVW = -newUVW;

      std::cout << "Old " << oldUVW << " (" << length(oldUVW) << ")\n";
      std::cout << "New " << newUVW << " (" << length(newUVW) << ")\n\n";
    }
  }
}

void rotateToSun(casacore::MeasurementSet& set, int fieldIndex,
                       MSField& fieldTable, bool onlyUVW, bool flipUVWSign) {
  aocommon::MultiBandData bandData(set.spectralWindow(), set.dataDescription());
  ScalarColumn<casacore::String> nameCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::NAME));
  MDirection::ArrayColumn phaseDirCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::PHASE_DIR));
  MEpoch::ScalarColumn timeCol(set, set.columnName(MSMainEnums::TIME));

  ScalarColumn<int> antenna1Col(set, set.columnName(MSMainEnums::ANTENNA1)),
      antenna2Col(set, set.columnName(MSMainEnums::ANTENNA2)),
      fieldIdCol(set, set.columnName(MSMainEnums::FIELD_ID)),
      dataDescIdCol(set, set.columnName(MSMainEnums::DATA_DESC_ID));
  Muvw::ScalarColumn uvwCol(set, set.columnName(MSMainEnums::UVW));
  ArrayColumn<double> uvwOutCol(set, set.columnName(MSMainEnums::UVW));

  const bool hasCorrData = set.isColumn(casacore::MSMainEnums::CORRECTED_DATA),
             hasModelData = set.isColumn(casacore::MSMainEnums::MODEL_DATA);
  std::unique_ptr<ArrayColumn<casacore::Complex>> dataCol, correctedDataCol,
      modelDataCol;
  if (!onlyUVW) {
    dataCol.reset(new ArrayColumn<casacore::Complex>(
        set, set.columnName(MSMainEnums::DATA)));

    if (hasCorrData) {
      correctedDataCol.reset(new ArrayColumn<casacore::Complex>(
          set, set.columnName(MSMainEnums::CORRECTED_DATA)));
    }
    if (hasModelData) {
      modelDataCol.reset(new ArrayColumn<casacore::Complex>(
          set, set.columnName(MSMainEnums::MODEL_DATA)));
    }
  }

  casacore::Vector<MDirection> phaseDirVector = phaseDirCol(fieldIndex);
  MDirection phaseDirection = phaseDirVector[0];
  double oldRA = phaseDirection.getAngle().getValue()[0];
  double oldDec = phaseDirection.getAngle().getValue()[1];

  casacore::IPosition dataShape;
  unsigned polarizationCount = 0;
  std::unique_ptr<casacore::Array<casacore::Complex>> dataArray;
  if (!onlyUVW) {
    dataShape = dataCol->shape(0);
    polarizationCount = dataShape[0];
    dataArray.reset(new casacore::Array<casacore::Complex>(dataShape));
  }

  std::unique_ptr<ProgressBar> progressBar;

  std::vector<Muvw> uvws(antennas.size());
  MEpoch time(MVEpoch(-1.0));
  for (unsigned row = 0; row != set.nrow(); ++row) {
    if (fieldIdCol(row) == fieldIndex) {
      // Read values from set
      const int antenna1 = antenna1Col(row), antenna2 = antenna2Col(row),
                dataDescId = dataDescIdCol(row);
      const Muvw oldUVW = uvwCol(row);

      MEpoch rowTime = timeCol(row);
      if (rowTime.getValue() != time.getValue()) {
        time = rowTime;

        MDirection newDirection = SolarDirection(set, row);
        double newRA = newDirection.getAngle().getValue()[0];
        double newDec = newDirection.getAngle().getValue()[1];
        double mjd = rowTime.getValue().get();
        time_t unix_time = static_cast<time_t>((mjd - 40587.0) * 86400.0);  // 40587 = MJD at Unix epoch
        std::tm* gm_time = std::gmtime(&unix_time);

        std::ostringstream isoTime;
        isoTime << std::setfill('0') << std::setw(4) << (gm_time->tm_year + 1900) << "-"
        << std::setw(2) << (gm_time->tm_mon + 1) << "-"
        << std::setw(2) << gm_time->tm_mday << "T"
        << std::setw(2) << gm_time->tm_hour << ":"
        << std::setw(2) << gm_time->tm_min << ":"
        << std::setw(2) << gm_time->tm_sec;

        std::cout << "Processing timestep in field \"" << nameCol(fieldIndex)
          << "\" at UTC " << isoTime.str() << " (MJD: " << mjd << "): "
          << dirToString(phaseDirection) << " -> "
          << dirToString(newDirection) << " ("
          << aocommon::ImageCoordinates::AngularDistance(oldRA, oldDec, newRA, newDec) *
                 (180.0 / M_PI)
          << " deg)\n";
        MDirection refDirection = MDirection::Convert(
            newDirection, MDirection::Ref(MDirection::J2000))();
        double dl, dm;
        aocommon::ImageCoordinates::RaDecToLM(oldRA, oldDec, newRA, newDec, dl,
                                              dm);

        for (size_t a = 0; a != antennas.size(); ++a)
          uvws[a] = calculateUVW(antennas[a], antennas[0], time, refDirection);
      }

      // Calculate the new UVW
      MVuvw newUVW = uvws[antenna1].getValue() - uvws[antenna2].getValue();
      if (flipUVWSign) newUVW = -newUVW;

      // If one of the first results, output values for analyzing them.
      if (row < 5) {
        std::cout << "Old " << oldUVW << " (" << length(oldUVW) << ")\n";
        std::cout << "New " << newUVW << " (" << length(newUVW) << ")\n\n";
      } else {
        if (!progressBar)
          progressBar.reset(new ProgressBar("Changing phase centre"));
        progressBar->SetProgress(row, set.nrow());
      }

      // Read the visibilities and phase-rotate them
      double shiftFactor =
          -2.0 * M_PI *
          (newUVW.getVector()[2] - oldUVW.getValue().getVector()[2]);

      if (!onlyUVW) {
        const aocommon::BandData& thisBand = bandData[dataDescId];
        dataCol->get(row, *dataArray);
        RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                           dataArray->cbegin());
        dataCol->put(row, *dataArray);

        if (hasCorrData) {
          correctedDataCol->get(row, *dataArray);
          RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                             dataArray->cbegin());
          correctedDataCol->put(row, *dataArray);
        }

        if (hasModelData) {
          modelDataCol->get(row, *dataArray);
          RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                             dataArray->cbegin());
          modelDataCol->put(row, *dataArray);
        }
      }

      // Store uvws
      uvwOutCol.put(row, newUVW.getVector());
    }
  }

  phaseDirVector[0] = SolarDirection(set, 0);
  phaseDirCol.put(fieldIndex, phaseDirVector);
}

void rotateToGeoZenith(casacore::MeasurementSet& set, int fieldIndex,
                       MSField& fieldTable, bool onlyUVW, bool flipUVWSign) {
  aocommon::MultiBandData bandData(set.spectralWindow(), set.dataDescription());
  ScalarColumn<casacore::String> nameCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::NAME));
  MDirection::ArrayColumn phaseDirCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::PHASE_DIR));
  MEpoch::ScalarColumn timeCol(set, set.columnName(MSMainEnums::TIME));

  ScalarColumn<int> antenna1Col(set, set.columnName(MSMainEnums::ANTENNA1)),
      antenna2Col(set, set.columnName(MSMainEnums::ANTENNA2)),
      fieldIdCol(set, set.columnName(MSMainEnums::FIELD_ID)),
      dataDescIdCol(set, set.columnName(MSMainEnums::DATA_DESC_ID));
  Muvw::ScalarColumn uvwCol(set, set.columnName(MSMainEnums::UVW));
  ArrayColumn<double> uvwOutCol(set, set.columnName(MSMainEnums::UVW));

  const bool hasCorrData = set.isColumn(casacore::MSMainEnums::CORRECTED_DATA),
             hasModelData = set.isColumn(casacore::MSMainEnums::MODEL_DATA);
  std::unique_ptr<ArrayColumn<casacore::Complex>> dataCol, correctedDataCol,
      modelDataCol;
  if (!onlyUVW) {
    dataCol.reset(new ArrayColumn<casacore::Complex>(
        set, set.columnName(MSMainEnums::DATA)));

    if (hasCorrData) {
      correctedDataCol.reset(new ArrayColumn<casacore::Complex>(
          set, set.columnName(MSMainEnums::CORRECTED_DATA)));
    }
    if (hasModelData) {
      modelDataCol.reset(new ArrayColumn<casacore::Complex>(
          set, set.columnName(MSMainEnums::MODEL_DATA)));
    }
  }

  casacore::Vector<MDirection> phaseDirVector = phaseDirCol(fieldIndex);
  MDirection phaseDirection = phaseDirVector[0];
  double oldRA = phaseDirection.getAngle().getValue()[0];
  double oldDec = phaseDirection.getAngle().getValue()[1];

  casacore::IPosition dataShape;
  unsigned polarizationCount = 0;
  std::unique_ptr<casacore::Array<casacore::Complex>> dataArray;
  if (!onlyUVW) {
    dataShape = dataCol->shape(0);
    polarizationCount = dataShape[0];
    dataArray.reset(new casacore::Array<casacore::Complex>(dataShape));
  }

  std::unique_ptr<ProgressBar> progressBar;

  std::vector<Muvw> uvws(antennas.size());
  MEpoch time(MVEpoch(-1.0));
  for (unsigned row = 0; row != set.nrow(); ++row) {
    if (fieldIdCol(row) == fieldIndex) {
      // Read values from set
      const int antenna1 = antenna1Col(row), antenna2 = antenna2Col(row),
                dataDescId = dataDescIdCol(row);
      const Muvw oldUVW = uvwCol(row);

      MEpoch rowTime = timeCol(row);
      if (rowTime.getValue() != time.getValue()) {
        time = rowTime;

        MDirection newDirection = ZenithDirection(set, row);
        double newRA = newDirection.getAngle().getValue()[0];
        double newDec = newDirection.getAngle().getValue()[1];
        std::cout << "Processing timestep in field \"" << nameCol(fieldIndex)
                  << "\": " << dirToString(phaseDirection) << " -> "
                  << dirToString(newDirection) << " ("
                  << aocommon::ImageCoordinates::AngularDistance(
                         oldRA, oldDec, newRA, newDec) *
                         (180.0 / M_PI)
                  << " deg)\n";
        MDirection refDirection = MDirection::Convert(
            newDirection, MDirection::Ref(MDirection::J2000))();
        double dl, dm;
        aocommon::ImageCoordinates::RaDecToLM(oldRA, oldDec, newRA, newDec, dl,
                                              dm);

        for (size_t a = 0; a != antennas.size(); ++a)
          uvws[a] = calculateUVW(antennas[a], antennas[0], time, refDirection);
      }

      // Calculate the new UVW
      MVuvw newUVW = uvws[antenna1].getValue() - uvws[antenna2].getValue();
      if (flipUVWSign) newUVW = -newUVW;

      // If one of the first results, output values for analyzing them.
      if (row < 5) {
        std::cout << "Old " << oldUVW << " (" << length(oldUVW) << ")\n";
        std::cout << "New " << newUVW << " (" << length(newUVW) << ")\n\n";
      } else {
        if (!progressBar)
          progressBar.reset(new ProgressBar("Changing phase centre"));
        progressBar->SetProgress(row, set.nrow());
      }

      // Read the visibilities and phase-rotate them
      double shiftFactor =
          -2.0 * M_PI *
          (newUVW.getVector()[2] - oldUVW.getValue().getVector()[2]);

      if (!onlyUVW) {
        const aocommon::BandData& thisBand = bandData[dataDescId];
        dataCol->get(row, *dataArray);
        RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                           dataArray->cbegin());
        dataCol->put(row, *dataArray);

        if (hasCorrData) {
          correctedDataCol->get(row, *dataArray);
          RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                             dataArray->cbegin());
          correctedDataCol->put(row, *dataArray);
        }

        if (hasModelData) {
          modelDataCol->get(row, *dataArray);
          RotateVisibilities(thisBand, shiftFactor, polarizationCount,
                             dataArray->cbegin());
          modelDataCol->put(row, *dataArray);
        }
      }

      // Store uvws
      uvwOutCol.put(row, newUVW.getVector());
    }
  }

  phaseDirVector[0] = ZenithDirection(set);
  phaseDirCol.put(fieldIndex, phaseDirVector);
}

void readAntennas(casacore::MeasurementSet& set,
                  std::vector<MPosition>& antennas) {
  MSAntenna antennaTable = set.antenna();
  MPosition::ScalarColumn posCol(
      antennaTable, antennaTable.columnName(MSAntennaEnums::POSITION));

  antennas.resize(antennaTable.nrow());
  for (unsigned i = 0; i != antennaTable.nrow(); ++i) {
    antennas[i] = MPosition::Convert(posCol(i), MPosition::ITRF)();
  }
}

MDirection MinWDirection(casacore::MeasurementSet& set) {
  MPosition centroid = ArrayCentroid(set);
  casacore::Vector<casacore::Double> cvec = centroid.getValue().getVector();
  double cx = cvec[0], cy = cvec[1], cz = cvec[2];

  casacore::MSAntenna aTable = set.antenna();
  casacore::MPosition::ScalarColumn antPosColumn(
      aTable, aTable.columnName(casacore::MSAntennaEnums::POSITION));
  integer n = 3, m = aTable.nrow(), lda = m, ldu = m, ldvt = n;
  std::vector<double> a(m * n);

  for (size_t row = 0; row != aTable.nrow(); ++row) {
    MPosition pos = antPosColumn(row);
    casacore::Vector<casacore::Double> vec = pos.getValue().getVector();
    a[row] = vec[0] - cx, a[row + m] = vec[1] - cy,
    a[row + 2 * m] = vec[2] - cz;
  }

  double wkopt;
  std::vector<double> s(n), u(ldu * m), vt(ldvt * n);
  integer lwork = -1, info = 0;
  dgesvd_("All", "All", &m, &n, &a[0], &lda, &s[0], &u[0], &ldu, &vt[0], &ldvt,
          &wkopt, &lwork, &info);
  lwork = (int)wkopt;
  double* work = (double*)malloc(lwork * sizeof(double));
  /* Compute SVD */
  dgesvd_("All", "All", &m, &n, &a[0], &lda, &s[0], &u[0], &ldu, &vt[0], &ldvt,
          work, &lwork, &info);
  free((void*)work);

  if (info > 0)
    throw std::runtime_error("The algorithm computing SVD failed to converge");
  else {
    // Get the right singular vector belonging to the smallest SV
    double x = vt[n * 0 + n - 1], y = vt[n * 1 + n - 1], z = vt[n * 2 + n - 1];
    // Get the hemisphere right
    if ((z < 0.0 && cz > 0.0) || (z > 0.0 && cz < 0.0)) {
      x = -x;
      y = -y;
      z = -z;
    }

    casacore::MEpoch::ScalarColumn timeColumn(
        set, set.columnName(casacore::MSMainEnums::TIME));
    casacore::MEpoch time = timeColumn(set.nrow() / 2);
    casacore::MeasFrame frame(centroid, time);
    MDirection::Ref ref(casacore::MDirection::ITRF, frame);
    casacore::MDirection direction(casacore::MVDirection(x, y, z), ref);
    const casacore::MDirection::Ref j2000Ref(casacore::MDirection::J2000,
                                             frame);
    return casacore::MDirection::Convert(direction, j2000Ref)();
  }
}

void printPhaseDir(casacore::MeasurementSet& set) {
  MSField fieldTable = set.field();
  MDirection::ArrayColumn phaseDirCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::PHASE_DIR));
  ScalarColumn<casacore::String> nameCol(
      fieldTable, fieldTable.columnName(MSFieldEnums::NAME));
  MDirection zenith = ZenithDirection(set);

  std::cout << "Current phase direction:\n";
  double dl, dm;
  getShift(fieldTable, dl, dm);
  for (size_t i = 0; i != fieldTable.nrow(); ++i) {
    casacore::Vector<MDirection> phaseDirVector = phaseDirCol(i);
    MDirection phaseDirection = phaseDirVector[0];
    std::cout << "  ";
    if (fieldTable.nrow() > 1) std::cout << nameCol(i) << " ";
    std::cout << dirToString(phaseDirection);
    if (dl != 0.0 || dm != 0.0) std::cout << ", shift: " << dl << ',' << dm;
    std::cout << '\n';
  }

  std::cout << "Zenith is at:\n  " << dirToString(zenith) << " ("
            << dirToString(ZenithDirectionStart(set)) << " - "
            << dirToString(ZenithDirectionEnd(set)) << ")\n";
  std::cout << "Min-w direction is at:\n  " << dirToString(MinWDirection(set))
            << '\n';
}

int main(int argc, char** argv) {
  std::cout << "A program to change the phase centre of a measurement set.\n"
               "Written by André Offringa (offringa@gmail.com).\n\n";
  if (argc < 2) {
    std::cout
        << "Syntax: chgcentre [options] <ms> <new ra> <new dec>\n\n"
           "The format of RA can either be 00h00m00.0s or 00:00:00.0\n"
           "The format of Dec can either be 00d00m00.0s or 00.00.00.0\n\n"
           "Example to rotate to HydA:\n"
           "\tchgcentre myset.ms 09h18m05.8s -12d05m44s\n\n"
           "Some options:\n"
           "-geozenith\n"
           "\tWill calculate the RA,dec of zenith for each timestep, and moves "
           "there. This make the set non-standard.\n"
           "-solarcenter\n"
           "\tWill calculate the RA,dec of solar center for each timestep, and moves "
           "there. This make the set non-standard.\n"
           "-flipuvwsign\n"
           "\tFlips the UVW sign. Necessary for LOFAR, for unknown reasons.\n"
           "-minw\n"
           "\tCalculate the direction that gives the minimum w-values for the "
           "array.\n"
           "-zenith\n"
           "\tShift to the average zenith value.\n"
           "-only-uvw\n"
           "\tOnly update UVW values, do not apply the phase shift.\n"
           "-shiftback\n"
           "\tAfter changing the phase centre, project the visibilities back "
           "to the old phase centre. This is useful\n"
           "\tin WSClean for imaging with minimum w-values in a different "
           "projection.\n"
           "-f\n"
           "\tForce recalculation, even if destination is same as original "
           "phase direction.\n"
           "-datacolumn <name>\n"
           "\tOnly phase-rotate the visibilities in the given column. "
           "Otherwise, the columns\n"
           "\tDATA, MODEL_DATA and CORRECTED_DATA will all be processed if "
           "they exist.\n"
           "-from-ms <ms>\n"
           "\tRotate the measurement set to the same direction as specified\n"
           "\tin the provided measurement set.\n"
           "\n";
  } else {
    int argi = 1;
    bool toZenith = false, toMinW = false, onlyUVW = false, shiftback = false,
         toGeozenith = false, toSun = false, flipUVWSign = false, force = false, show = false,
         same = false;
    double newDl = 0.0, newDm = 0.0;
    std::string templateMS;
    std::string dataColumn;
    while (argv[argi][0] == '-') {
      std::string param(&argv[argi][1]);
      if (param == "zenith") {
        toZenith = true;
      } else if (param == "geozenith") {
        toGeozenith = true;
      } else if (param == "solarcenter") {
        toSun = true;
      } else if (param == "minw") {
        toMinW = true;
      } else if (param == "only-uvw") {
        onlyUVW = true;
      } else if (param == "shiftback") {
        shiftback = true;
      } else if (param == "flipuvwsign") {
        flipUVWSign = true;
      } else if (param == "f") {
        force = true;
      } else if (param == "show") {
        show = true;
      } else if (param == "same") {
        same = true;
      } else if (param == "datacolumn") {
        ++argi;
        dataColumn = argv[argi];
      } else if (param == "from-ms") {
        ++argi;
        templateMS = argv[argi];
      } else
        throw std::runtime_error("Invalid parameter");
      ++argi;
    }
    if (argi == argc)
      std::cout << "Missing parameter.\n";
    else if (argi + 1 == argc && !toZenith && !toMinW && !toGeozenith && !toSun &&
             !same && templateMS.empty()) {
      casacore::MeasurementSet set(argv[argi]);
      readAntennas(set, antennas);
      printPhaseDir(set);
    } else {
      casacore::MeasurementSet set;
      if (show)
        set = casacore::MeasurementSet(argv[argi]);
      else
        set = casacore::MeasurementSet(argv[argi], casacore::Table::Update);
      readAntennas(set, antennas);
      MDirection newDirection;
      if (toZenith) {
        newDirection = ZenithDirection(set);
      } else if (toMinW) {
        newDirection = MinWDirection(set);
      } else if (same) {
        MDirection::ArrayColumn phaseDirCol(
            set.field(), set.field().columnName(MSFieldEnums::PHASE_DIR));
        casacore::Vector<MDirection> phaseDirVector = phaseDirCol(0);
        newDirection = phaseDirVector[0];
      } else if (!templateMS.empty()) {
        casacore::MeasurementSet fromMS(templateMS);
        MSField fromField(fromMS.field());
        MDirection::ArrayColumn phaseDirCol(
            fromField, fromField.columnName(MSFieldEnums::PHASE_DIR));
        casacore::Vector<MDirection> phaseDirVector = phaseDirCol(0);
        newDirection = phaseDirVector[0];
        getShift(fromField, newDl, newDm);
      } else if (!toGeozenith && !toSun && argc >= argi + 3) {
         double newRA = RaDecCoord::ParseRA(argv[argi + 1]);
         double newDec = RaDecCoord::ParseDec(argv[argi + 2]);
         newDirection = MDirection(MVDirection(newRA, newDec),
                              MDirection::Ref(MDirection::J2000));
      } else if (toSun) {
      }

      MSField fieldTable = set.field();
      for (unsigned fieldIndex = 0; fieldIndex != fieldTable.nrow();
           ++fieldIndex) {
        if (show)
          showChanges(set, fieldIndex, fieldTable, newDirection, flipUVWSign);
        else if (toGeozenith)
          rotateToGeoZenith(set, fieldIndex, fieldTable, onlyUVW, flipUVWSign);
	else if (toSun)
          rotateToSun(set, fieldIndex, fieldTable, onlyUVW, flipUVWSign);
        else
          processField(set, dataColumn, fieldIndex, fieldTable, newDirection,
                       onlyUVW, shiftback, newDl, newDm, flipUVWSign, force);
      }
    }
  }

  return 0;
}

