import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.processing.edges.SUSANEdgeDetector;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;
import spark.Request;
import spark.Route;

import javax.servlet.MultipartConfigElement;
import javax.servlet.ServletException;
import javax.servlet.http.Part;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

import java.nio.file.Paths;
import java.net.URL;

import org.apache.commons.lang3.StringUtils;
import java.io.File;

import java.net.URISyntaxException;

import static spark.Spark.*;
import static spark.debug.DebugScreen.enableDebugScreen;

public class Main {

    private static final ClassLoader classLoader = Main.class.getClassLoader();
    public static final String PNG = ".png";

    private static final String BASE_PATH = getBasePath();
    private static final String IMAGES_PATH = BASE_PATH + "/images";
    private static final String DAT_FILE_PATH = BASE_PATH + "/training_data_files";
    private static final File UPLOAD_DIRECTORY = new File("upload");

    private static final Route formRoute = (req, res) ->
            "<form method='post' enctype='multipart/form-data'>" // note the enctype
                    + "    <input type='file' name='uploaded_file' accept='.png'>" // make sure to call getPart using the same "name" in the post
                    + "    <button>Upload picture</button>"
                    + "</form>";


    private static final String ROUTE_PART0 = "/header";
    private static final String TRAINING_IMAGES_PATH_PART0 = IMAGES_PATH + "/dataset_part0";
    private static final String TRAINER_DATA_FILE_PATH_PART0 = DAT_FILE_PATH + "/trainer_part0.dat";
    private static final int PART0_NUMGROUP = 2;
    private static final int PART0_NUMTRAINING = 4;
    private static final int PART0_NUMTESTING = 4;
    private static final int PART0_NUMBER = 7;


    private static final String ROUTE_PART1 = "/stage";
    private static final String TRAINING_IMAGES_PATH_PART1 = IMAGES_PATH + "/dataset_part1";
    private static final String TRAINER_DATA_FILE_PATH_PART1 = DAT_FILE_PATH + "/trainer_part1.dat";
    private static final int PART1_NUMGROUP = 3;
    private static final int PART1_NUMTRAINING = 5;
    private static final int PART1_NUMTESTING = 5;
    private static final int PART1_NUMBER = 9;

    private static final String ROUTE_PART2 = "/teaserlist";
    private static final String TRAINING_IMAGES_PATH_PART2 = IMAGES_PATH + "/dataset_part2";
    private static final String TRAINER_DATA_FILE_PATH_PART2 = DAT_FILE_PATH + "/trainer_part2.dat";
    private static final int PART2_NUMGROUP = 3;
    private static final int PART2_NUMTRAINING = 10;
    private static final int PART2_NUMTESTING = 10;
    private static final int PART2_NUMBER = 15;

    public static String getBasePath() {
        String basePath = "../src/main/resources/sketch_backup";

//        URL resource = Main.class.getResource("sketch_backup/path.txt");
//        System.out.println(resource);
//
//        try {
//            File tempFile = Paths.get(resource.toURI()).toFile();
//            String filePath = tempFile.getAbsolutePath();
//            basePath = StringUtils.removeEnd(filePath, "/path.txt");
//        } catch (URISyntaxException e) {
//            e.printStackTrace();
//        }
        return basePath;
    }

    public static final ConfigObject[] CONFIG_ARRAYS = {
            new ConfigObject(ROUTE_PART0,
                    TRAINING_IMAGES_PATH_PART0,
                    TRAINER_DATA_FILE_PATH_PART0,
                    PART0_NUMGROUP,
                    PART0_NUMTRAINING,
                    PART0_NUMTESTING,
                    PART0_NUMBER),

            new ConfigObject(ROUTE_PART1,
                    TRAINING_IMAGES_PATH_PART1,
                    TRAINER_DATA_FILE_PATH_PART1,
                    PART1_NUMGROUP,
                    PART1_NUMTRAINING,
                    PART1_NUMTESTING,
                    PART1_NUMBER),

            new ConfigObject(ROUTE_PART2,
                    TRAINING_IMAGES_PATH_PART2,
                    TRAINER_DATA_FILE_PATH_PART2,
                    PART2_NUMGROUP,
                    PART2_NUMTRAINING,
                    PART2_NUMTESTING,
                    PART2_NUMBER),
    };

    static int getHerokuAssignedPort() {
        ProcessBuilder processBuilder = new ProcessBuilder();
        if (processBuilder.environment().get("PORT") != null) {
            return Integer.parseInt(processBuilder.environment().get("PORT"));
        }
        return 4567; //return default port if heroku-port isn't set (i.e. on localhost)
    }

    public static void main(String[] args) {
        port(getHerokuAssignedPort());

        enableDebugScreen();

        UPLOAD_DIRECTORY.mkdir(); // create the upload directory if it doesn't exist

        staticFiles.externalLocation("upload");
        staticFiles.location("/public");

        for (int i = 0; i < CONFIG_ARRAYS.length; i++) {
            setupRoutes(CONFIG_ARRAYS[i]);
        }

        enableCORS("*", "GET, PUT, POST, DELETE, HEAD", "*");
    }

    private static void enableCORS(final String origin, final String methods, final String headers) {

        options("/*",
                (request, response) -> {

                    String accessControlRequestHeaders = request
                            .headers("Access-Control-Request-Headers");
                    if (accessControlRequestHeaders != null) {
                        response.header("Access-Control-Allow-Headers",
                                accessControlRequestHeaders);
                    }

                    String accessControlRequestMethod = request
                            .headers("Access-Control-Request-Method");
                    if (accessControlRequestMethod != null) {
                        response.header("Access-Control-Allow-Methods",
                                accessControlRequestMethod);
                    }

                    return "OK";
                });

        before((request, response) -> response.header("Access-Control-Allow-Origin", "*"));
    }


    private static void setupRoutes(ConfigObject currentConfig) {
        get(currentConfig.getRoute(), formRoute
        );

        post(currentConfig.getRoute(), (req, res) -> {

            Path tempFile = Files.createTempFile(UPLOAD_DIRECTORY.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));


            LiblinearAnnotator<FImage, String> trainer = null;
            File inputDataFile = new File(currentConfig.getTrainerDataFile());                                           //TRAINER_DATA_FILE_PATH_PART0
            if (inputDataFile.isFile()) {
                trainer = IOUtils.readFromFile(inputDataFile);
            } else {
                preProcessImages(currentConfig);

                VFSGroupDataset<FImage> allData = null;
                allData = new VFSGroupDataset<FImage>(
                        currentConfig.getTrainingImagesPath(),                                                                 //TRAINING_IMAGES_PATH_PART0
                        ImageUtilities.FIMAGE_READER);

                GroupedDataset<String, ListDataset<FImage>, FImage> data =
                        GroupSampler.sample(allData, currentConfig.getNumGroup(), false);

                GroupedRandomSplitter<String, FImage> splits =
                        new GroupedRandomSplitter<String, FImage>(data, currentConfig.getNumTraining(), 0, currentConfig.getNumTesting()); // 15 training, 15 testing


                DenseSIFT denseSIFT = new DenseSIFT(5, 10);
                PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<FImage>(denseSIFT, 6f, 10);

                GroupedDataset<String, ListDataset<FImage>, FImage> sample =
                        GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), currentConfig.getNumber());

                HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(sample, pyramidDenseSIFT);

                FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pyramidDenseSIFT, assigner);

                //
                // Now we’re ready to construct and train a classifier
                //
                trainer = new LiblinearAnnotator<FImage, String>(
                        extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

                Date start = new Date();
                System.out.println("Classifier training: start");
                trainer.train(splits.getTrainingDataset());
                File f = new File(currentConfig.getTrainerDataFile());                                                //TRAINER_DATA_FILE_PATH_PART0
                if (!f.getParentFile().exists())
                    f.getParentFile().mkdirs();
                if (!f.exists())
                    f.createNewFile();

                IOUtils.writeToFile(trainer, new File(currentConfig.getTrainerDataFile()));                               //TRAINER_DATA_FILE_PATH_PART0
                System.out.println("Classifier training: end");
                Date end = new Date();
                long durationSec = (end.getTime() - start.getTime()) / 1000;
                System.out.println("Classifier training duration: " + durationSec + " seconds");
            }

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);


                FImage query = ImageUtilities.readF(tempFile.toFile());
                query = SUSANEdgeDetector.smoothCircularSusan( query, 0.01, 4, 3.4 );

                final List<ScoredAnnotation<String>> scoredAnnotations = trainer.annotate(query);
                final ClassificationResult<String> classificationResult = trainer.classify(query);
                System.out.println("scoredAnnotations: " + scoredAnnotations);
                System.out.println("classificationResult: " + classificationResult.getPredictedClasses());
                return "{ \"endpoint\": \""+ currentConfig.getRoute().substring(1) +"\", \"classification\":\""+ classificationResult.getPredictedClasses().iterator().next() + "\"}";

            }

        });
    }


    private static void preProcessImages(ConfigObject currentConfig) throws IOException {
        VFSGroupDataset<FImage> imageDataset = null;
        imageDataset = new VFSGroupDataset<FImage>(
                currentConfig.getTrainingImagesPath(),
                ImageUtilities.FIMAGE_READER);

        for (final Map.Entry<String, VFSListDataset<FImage>> entry : imageDataset.entrySet()) {
            String folderPath = currentConfig.getTrainingImagesPath() + File.separator + entry.getKey();

            int i = 0;
            for (FImage image : entry.getValue()){
                String imagePath = folderPath + File.separator + i + PNG;
                ImageUtilities.write(SUSANEdgeDetector.smoothCircularSusan( image, 0.01, 4, 3.4 ), new File(imagePath));
                i++;
            }
            clearOldFilesFromFolder(folderPath);
        }
    }

    private static void clearOldFilesFromFolder(String folderPath){
        //Clear the folder before pre-processing
        File folder = new File(folderPath);
        File[] files = folder.listFiles();
        if(files!=null) { //some JVMs return null for empty dirs
            for(File f: files) {
                String filePath = f.getAbsolutePath();
                if (filePath.substring(filePath.lastIndexOf(File.separator)).contains("part")) {
                    f.delete();
                }
            }
        }
    }
    // methods used for logging
    private static void logInfo(Request req, Path tempFile) throws IOException, ServletException {
        System.out.println("Uploaded file '" + getFileName(req.raw().getPart("uploaded_file")) + "' saved as '" + tempFile.toAbsolutePath() + "'");
    }

    private static String getFileName(Part part) {
        for (String cd : part.getHeader("content-disposition").split(";")) {
            if (cd.trim().startsWith("filename")) {
                return cd.substring(cd.indexOf('=') + 1).trim().replace("\"", "");
            }
        }
        return null;
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> sample,
            PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
        System.out.println("trainQuantiser: start");
        Date start = new Date();
        List<LocalFeatureList<ByteDSIFTKeypoint>> allKeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        int i = 0;

        int total = sample.numInstances();
        for (FImage rec : sample) {
            i++;
            System.out.println(String.format("Analysing image %d out of %d", i, total));
            FImage img = rec.getImage();

            pyramidDenseSIFT.analyseImage(img);
            allKeys.add(pyramidDenseSIFT.getByteKeypoints(0.005f));
        }
        final int numberOfDenseSiftFeaturesToExtract = 1000;
        final int numberOfClassesInCluster = 100;
        if (allKeys.size() > numberOfDenseSiftFeaturesToExtract)
            allKeys = allKeys.subList(0, numberOfDenseSiftFeaturesToExtract);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(numberOfClassesInCluster);
        DataSource<byte[]> dataSource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allKeys);
        System.out.println(String.format(
                "Clustering %d image features into %d classes...",
                numberOfDenseSiftFeaturesToExtract, numberOfClassesInCluster));
        ByteCentroidsResult result = km.cluster(dataSource);
        Date end = new Date();
        System.out.println("trainQuantiser: end");
        System.out.println("trainQuantiser duration: " + (end.getTime() - start.getTime()) / 1000 + " seconds");
        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}