public class ConfigObject {
    private String route;
    private String trainingImagesPath;
    private String trainerDataFile;

    public ConfigObject(String route, String trainingImagesPath, String trainerDataFile) {
        this.route = route;
        this.trainingImagesPath = trainingImagesPath;
        this.trainerDataFile = trainerDataFile;
    }

    public String getRoute() {
        return route;
    }

    public String getTrainingImagesPath() {
        return trainingImagesPath;
    }

    public String getTrainerDataFile() {
        return trainerDataFile;
    }
}
