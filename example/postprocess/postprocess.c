#include "postprocess.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <stdbool.h>

#ifdef HARDWARE_DRAW
 #include "imageScaler/scaler.h"
 #include "detectionDemo.h"
 #include "frameDrawing/draw_assist.h"
 #include "frameDrawing/draw.h"
 #include <fcntl.h>
 static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
 extern int fps;
 extern int update_Classifier;
 extern uint32_t* overlay_draw_frame;

 void privacy_draw(int split){
	for(int s =0; s< split; s++){
		draw_dma_memset(1920,1080/split, overlay_draw_frame + (s*1080/split)*2048, 2048, GET_COLOUR(0,0,0,255));
	}
 }
 void pixel_draw(model_t *model, vbx_cnn_io_ptr_t* o_buffers, vbx_cnn_t *the_vbx_cnn){
		int odims = model_get_output_dims(model,0);
		int *oshape = model_get_output_shape(model,0);
		int ow = oshape[odims-1];
		int oh = oshape[odims-2];
		uint32_t* output=(uint32_t*)(uintptr_t)o_buffers[0];
		//uint32_t* output=(uint32_t*)(uintptr_t)model_get_test_output(model,0);
		draw_dma_memcpy(ow,oh, overlay_draw_frame, 2048, virt_to_phys(the_vbx_cnn,output), ow);
 }
#else
	int update_Classifier=1;
#endif
static int topk_draw = 4;
static int16_t indexes[5] = {0};
static int16_t display_index[5] = {0};
static int32_t scores[5] = {0};

char *dota_classes[15] = {
  "plane",
  "ship",
  "storage tank",
  "baseball diamond",
  "tennis court",
  "basketball court",
  "ground track field",
  "harbor",
  "bridge",
  "large vehicle",
  "small vehicle",
  "helicopter",
  "roundabout",
  "soccer ball field",
  "swimming pool",
};

char *lpr_chinese_characters[71] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "Anhui", "Beijing", "Chongqing", "Fujian",
         "Gansu", "Guangdong", "Guangxi", "Guizhou",
         "Hainan", "Hebei", "Heilongjiang", "Henan",
         "HongKong", "Hubei", "Hunan", "InnerMongolia",
         "Jiangsu", "Jiangxi", "Jilin", "Liaoning",
         "Macau", "Ningxia", "Qinghai", "Shaanxi",
         "Shandong", "Shanghai", "Shanxi", "Sichuan",
         "Tianjin", "Tibet", "Xinjiang", "Yunnan",
         "Zhejiang", "Police",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z", " "};


char *lpr_characters[37] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z", " "};


char *imagenet_classes[1000] = {
	"tench",
	"goldfish",
	"great white shark",
	"tiger shark",
	"hammerhead",
	"electric ray",
	"stingray",
	"cock",
	"hen",
	"ostrich",
	"brambling",
	"goldfinch",
	"house finch",
	"junco",
	"indigo bunting",
	"robin",
	"bulbul",
	"jay",
	"magpie",
	"chickadee",
	"water ouzel",
	"kite",
	"bald eagle",
	"vulture",
	"great grey owl",
	"European fire salamander",
	"common newt",
	"eft",
	"spotted salamander",
	"axolotl",
	"bullfrog",
	"tree frog",
	"tailed frog",
	"loggerhead",
	"leatherback turtle",
	"mud turtle",
	"terrapin",
	"box turtle",
	"banded gecko",
	"common iguana",
	"American chameleon",
	"whiptail",
	"agama",
	"frilled lizard",
	"alligator lizard",
	"Gila monster",
	"green lizard",
	"African chameleon",
	"Komodo dragon",
	"African crocodile",
	"American alligator",
	"triceratops",
	"thunder snake",
	"ringneck snake",
	"hognose snake",
	"green snake",
	"king snake",
	"garter snake",
	"water snake",
	"vine snake",
	"night snake",
	"boa constrictor",
	"rock python",
	"Indian cobra",
	"green mamba",
	"sea snake",
	"horned viper",
	"diamondback",
	"sidewinder",
	"trilobite",
	"harvestman",
	"scorpion",
	"black and gold garden spider",
	"barn spider",
	"garden spider",
	"black widow",
	"tarantula",
	"wolf spider",
	"tick",
	"centipede",
	"black grouse",
	"ptarmigan",
	"ruffed grouse",
	"prairie chicken",
	"peacock",
	"quail",
	"partridge",
	"African grey",
	"macaw",
	"sulphur-crested cockatoo",
	"lorikeet",
	"coucal",
	"bee eater",
	"hornbill",
	"hummingbird",
	"jacamar",
	"toucan",
	"drake",
	"red-breasted merganser",
	"goose",
	"black swan",
	"tusker",
	"echidna",
	"platypus",
	"wallaby",
	"koala",
	"wombat",
	"jellyfish",
	"sea anemone",
	"brain coral",
	"flatworm",
	"nematode",
	"conch",
	"snail",
	"slug",
	"sea slug",
	"chiton",
	"chambered nautilus",
	"Dungeness crab",
	"rock crab",
	"fiddler crab",
	"king crab",
	"American lobster",
	"spiny lobster",
	"crayfish",
	"hermit crab",
	"isopod",
	"white stork",
	"black stork",
	"spoonbill",
	"flamingo",
	"little blue heron",
	"American egret",
	"bittern",
	"crane",
	"limpkin",
	"European gallinule",
	"American coot",
	"bustard",
	"ruddy turnstone",
	"red-backed sandpiper",
	"redshank",
	"dowitcher",
	"oystercatcher",
	"pelican",
	"king penguin",
	"albatross",
	"grey whale",
	"killer whale",
	"dugong",
	"sea lion",
	"Chihuahua",
	"Japanese spaniel",
	"Maltese dog",
	"Pekinese",
	"Shih-Tzu",
	"Blenheim spaniel",
	"papillon",
	"toy terrier",
	"Rhodesian ridgeback",
	"Afghan hound",
	"basset",
	"beagle",
	"bloodhound",
	"bluetick",
	"black-and-tan coonhound",
	"Walker hound",
	"English foxhound",
	"redbone",
	"borzoi",
	"Irish wolfhound",
	"Italian greyhound",
	"whippet",
	"Ibizan hound",
	"Norwegian elkhound",
	"otterhound",
	"Saluki",
	"Scottish deerhound",
	"Weimaraner",
	"Staffordshire bullterrier",
	"American Staffordshire terrier",
	"Bedlington terrier",
	"Border terrier",
	"Kerry blue terrier",
	"Irish terrier",
	"Norfolk terrier",
	"Norwich terrier",
	"Yorkshire terrier",
	"wire-haired fox terrier",
	"Lakeland terrier",
	"Sealyham terrier",
	"Airedale",
	"cairn",
	"Australian terrier",
	"Dandie Dinmont",
	"Boston bull",
	"miniature schnauzer",
	"giant schnauzer",
	"standard schnauzer",
	"Scotch terrier",
	"Tibetan terrier",
	"silky terrier",
	"soft-coated wheaten terrier",
	"West Highland white terrier",
	"Lhasa",
	"flat-coated retriever",
	"curly-coated retriever",
	"golden retriever",
	"Labrador retriever",
	"Chesapeake Bay retriever",
	"German short-haired pointer",
	"vizsla",
	"English setter",
	"Irish setter",
	"Gordon setter",
	"Brittany spaniel",
	"clumber",
	"English springer",
	"Welsh springer spaniel",
	"cocker spaniel",
	"Sussex spaniel",
	"Irish water spaniel",
	"kuvasz",
	"schipperke",
	"groenendael",
	"malinois",
	"briard",
	"kelpie",
	"komondor",
	"Old English sheepdog",
	"Shetland sheepdog",
	"collie",
	"Border collie",
	"Bouvier des Flandres",
	"Rottweiler",
	"German shepherd",
	"Doberman",
	"miniature pinscher",
	"Greater Swiss Mountain dog",
	"Bernese mountain dog",
	"Appenzeller",
	"EntleBucher",
	"boxer",
	"bull mastiff",
	"Tibetan mastiff",
	"French bulldog",
	"Great Dane",
	"Saint Bernard",
	"Eskimo dog",
	"malamute",
	"Siberian husky",
	"dalmatian",
	"affenpinscher",
	"basenji",
	"pug",
	"Leonberg",
	"Newfoundland",
	"Great Pyrenees",
	"Samoyed",
	"Pomeranian",
	"chow",
	"keeshond",
	"Brabancon griffon",
	"Pembroke",
	"Cardigan",
	"toy poodle",
	"miniature poodle",
	"standard poodle",
	"Mexican hairless",
	"timber wolf",
	"white wolf",
	"red wolf",
	"coyote",
	"dingo",
	"dhole",
	"African hunting dog",
	"hyena",
	"red fox",
	"kit fox",
	"Arctic fox",
	"grey fox",
	"tabby",
	"tiger cat",
	"Persian cat",
	"Siamese cat",
	"Egyptian cat",
	"cougar",
	"lynx",
	"leopard",
	"snow leopard",
	"jaguar",
	"lion",
	"tiger",
	"cheetah",
	"brown bear",
	"American black bear",
	"ice bear",
	"sloth bear",
	"mongoose",
	"meerkat",
	"tiger beetle",
	"ladybug",
	"ground beetle",
	"long-horned beetle",
	"leaf beetle",
	"dung beetle",
	"rhinoceros beetle",
	"weevil",
	"fly",
	"bee",
	"ant",
	"grasshopper",
	"cricket",
	"walking stick",
	"cockroach",
	"mantis",
	"cicada",
	"leafhopper",
	"lacewing",
	"dragonfly",
	"damselfly",
	"admiral",
	"ringlet",
	"monarch",
	"cabbage butterfly",
	"sulphur butterfly",
	"lycaenid",
	"starfish",
	"sea urchin",
	"sea cucumber",
	"wood rabbit",
	"hare",
	"Angora",
	"hamster",
	"porcupine",
	"fox squirrel",
	"marmot",
	"beaver",
	"guinea pig",
	"sorrel",
	"zebra",
	"hog",
	"wild boar",
	"warthog",
	"hippopotamus",
	"ox",
	"water buffalo",
	"bison",
	"ram",
	"bighorn",
	"ibex",
	"hartebeest",
	"impala",
	"gazelle",
	"Arabian camel",
	"llama",
	"weasel",
	"mink",
	"polecat",
	"black-footed ferret",
	"otter",
	"skunk",
	"badger",
	"armadillo",
	"three-toed sloth",
	"orangutan",
	"gorilla",
	"chimpanzee",
	"gibbon",
	"siamang",
	"guenon",
	"patas",
	"baboon",
	"macaque",
	"langur",
	"colobus",
	"proboscis monkey",
	"marmoset",
	"capuchin",
	"howler monkey",
	"titi",
	"spider monkey",
	"squirrel monkey",
	"Madagascar cat",
	"indri",
	"Indian elephant",
	"African elephant",
	"lesser panda",
	"giant panda",
	"barracouta",
	"eel",
	"coho",
	"rock beauty",
	"anemone fish",
	"sturgeon",
	"gar",
	"lionfish",
	"puffer",
	"abacus",
	"abaya",
	"academic gown",
	"accordion",
	"acoustic guitar",
	"aircraft carrier",
	"airliner",
	"airship",
	"altar",
	"ambulance",
	"amphibian",
	"analog clock",
	"apiary",
	"apron",
	"ashcan",
	"assault rifle",
	"backpack",
	"bakery",
	"balance beam",
	"balloon",
	"ballpoint",
	"Band Aid",
	"banjo",
	"bannister",
	"barbell",
	"barber chair",
	"barbershop",
	"barn",
	"barometer",
	"barrel",
	"barrow",
	"baseball",
	"basketball",
	"bassinet",
	"bassoon",
	"bathing cap",
	"bath towel",
	"bathtub",
	"beach wagon",
	"beacon",
	"beaker",
	"bearskin",
	"beer bottle",
	"beer glass",
	"bell cote",
	"bib",
	"bicycle-built-for-two",
	"bikini",
	"binder",
	"binoculars",
	"birdhouse",
	"boathouse",
	"bobsled",
	"bolo tie",
	"bonnet",
	"bookcase",
	"bookshop",
	"bottlecap",
	"bow",
	"bow tie",
	"brass",
	"brassiere",
	"breakwater",
	"breastplate",
	"broom",
	"bucket",
	"buckle",
	"bulletproof vest",
	"bullet train",
	"butcher shop",
	"cab",
	"caldron",
	"candle",
	"cannon",
	"canoe",
	"can opener",
	"cardigan",
	"car mirror",
	"carousel",
	"carpenter's kit",
	"carton",
	"car wheel",
	"cash machine",
	"cassette",
	"cassette player",
	"castle",
	"catamaran",
	"CD player",
	"cello",
	"cellular telephone",
	"chain",
	"chainlink fence",
	"chain mail",
	"chain saw",
	"chest",
	"chiffonier",
	"chime",
	"china cabinet",
	"Christmas stocking",
	"church",
	"cinema",
	"cleaver",
	"cliff dwelling",
	"cloak",
	"clog",
	"cocktail shaker",
	"coffee mug",
	"coffeepot",
	"coil",
	"combination lock",
	"computer keyboard",
	"confectionery",
	"container ship",
	"convertible",
	"corkscrew",
	"cornet",
	"cowboy boot",
	"cowboy hat",
	"cradle",
	"crane",
	"crash helmet",
	"crate",
	"crib",
	"Crock Pot",
	"croquet ball",
	"crutch",
	"cuirass",
	"dam",
	"desk",
	"desktop computer",
	"dial telephone",
	"diaper",
	"digital clock",
	"digital watch",
	"dining table",
	"dishrag",
	"dishwasher",
	"disk brake",
	"dock",
	"dogsled",
	"dome",
	"doormat",
	"drilling platform",
	"drum",
	"drumstick",
	"dumbbell",
	"Dutch oven",
	"electric fan",
	"electric guitar",
	"electric locomotive",
	"entertainment center",
	"envelope",
	"espresso maker",
	"face powder",
	"feather boa",
	"file",
	"fireboat",
	"fire engine",
	"fire screen",
	"flagpole",
	"flute",
	"folding chair",
	"football helmet",
	"forklift",
	"fountain",
	"fountain pen",
	"four-poster",
	"freight car",
	"French horn",
	"frying pan",
	"fur coat",
	"garbage truck",
	"gasmask",
	"gas pump",
	"goblet",
	"go-kart",
	"golf ball",
	"golfcart",
	"gondola",
	"gong",
	"gown",
	"grand piano",
	"greenhouse",
	"grille",
	"grocery store",
	"guillotine",
	"hair slide",
	"hair spray",
	"half track",
	"hammer",
	"hamper",
	"hand blower",
	"hand-held computer",
	"handkerchief",
	"hard disc",
	"harmonica",
	"harp",
	"harvester",
	"hatchet",
	"holster",
	"home theater",
	"honeycomb",
	"hook",
	"hoopskirt",
	"horizontal bar",
	"horse cart",
	"hourglass",
	"iPod",
	"iron",
	"jack-o'-lantern",
	"jean",
	"jeep",
	"jersey",
	"jigsaw puzzle",
	"jinrikisha",
	"joystick",
	"kimono",
	"knee pad",
	"knot",
	"lab coat",
	"ladle",
	"lampshade",
	"laptop",
	"lawn mower",
	"lens cap",
	"letter opener",
	"library",
	"lifeboat",
	"lighter",
	"limousine",
	"liner",
	"lipstick",
	"Loafer",
	"lotion",
	"loudspeaker",
	"loupe",
	"lumbermill",
	"magnetic compass",
	"mailbag",
	"mailbox",
	"maillot",
	"maillot",
	"manhole cover",
	"maraca",
	"marimba",
	"mask",
	"matchstick",
	"maypole",
	"maze",
	"measuring cup",
	"medicine chest",
	"megalith",
	"microphone",
	"microwave",
	"military uniform",
	"milk can",
	"minibus",
	"miniskirt",
	"minivan",
	"missile",
	"mitten",
	"mixing bowl",
	"mobile home",
	"Model T",
	"modem",
	"monastery",
	"monitor",
	"moped",
	"mortar",
	"mortarboard",
	"mosque",
	"mosquito net",
	"motor scooter",
	"mountain bike",
	"mountain tent",
	"mouse",
	"mousetrap",
	"moving van",
	"muzzle",
	"nail",
	"neck brace",
	"necklace",
	"nipple",
	"notebook",
	"obelisk",
	"oboe",
	"ocarina",
	"odometer",
	"oil filter",
	"organ",
	"oscilloscope",
	"overskirt",
	"oxcart",
	"oxygen mask",
	"packet",
	"paddle",
	"paddlewheel",
	"padlock",
	"paintbrush",
	"pajama",
	"palace",
	"panpipe",
	"paper towel",
	"parachute",
	"parallel bars",
	"park bench",
	"parking meter",
	"passenger car",
	"patio",
	"pay-phone",
	"pedestal",
	"pencil box",
	"pencil sharpener",
	"perfume",
	"Petri dish",
	"photocopier",
	"pick",
	"pickelhaube",
	"picket fence",
	"pickup",
	"pier",
	"piggy bank",
	"pill bottle",
	"pillow",
	"ping-pong ball",
	"pinwheel",
	"pirate",
	"pitcher",
	"plane",
	"planetarium",
	"plastic bag",
	"plate rack",
	"plow",
	"plunger",
	"Polaroid camera",
	"pole",
	"police van",
	"poncho",
	"pool table",
	"pop bottle",
	"pot",
	"potter's wheel",
	"power drill",
	"prayer rug",
	"printer",
	"prison",
	"projectile",
	"projector",
	"puck",
	"punching bag",
	"purse",
	"quill",
	"quilt",
	"racer",
	"racket",
	"radiator",
	"radio",
	"radio telescope",
	"rain barrel",
	"recreational vehicle",
	"reel",
	"reflex camera",
	"refrigerator",
	"remote control",
	"restaurant",
	"revolver",
	"rifle",
	"rocking chair",
	"rotisserie",
	"rubber eraser",
	"rugby ball",
	"rule",
	"running shoe",
	"safe",
	"safety pin",
	"saltshaker",
	"sandal",
	"sarong",
	"sax",
	"scabbard",
	"scale",
	"school bus",
	"schooner",
	"scoreboard",
	"screen",
	"screw",
	"screwdriver",
	"seat belt",
	"sewing machine",
	"shield",
	"shoe shop",
	"shoji",
	"shopping basket",
	"shopping cart",
	"shovel",
	"shower cap",
	"shower curtain",
	"ski",
	"ski mask",
	"sleeping bag",
	"slide rule",
	"sliding door",
	"slot",
	"snorkel",
	"snowmobile",
	"snowplow",
	"soap dispenser",
	"soccer ball",
	"sock",
	"solar dish",
	"sombrero",
	"soup bowl",
	"space bar",
	"space heater",
	"space shuttle",
	"spatula",
	"speedboat",
	"spider web",
	"spindle",
	"sports car",
	"spotlight",
	"stage",
	"steam locomotive",
	"steel arch bridge",
	"steel drum",
	"stethoscope",
	"stole",
	"stone wall",
	"stopwatch",
	"stove",
	"strainer",
	"streetcar",
	"stretcher",
	"studio couch",
	"stupa",
	"submarine",
	"suit",
	"sundial",
	"sunglass",
	"sunglasses",
	"sunscreen",
	"suspension bridge",
	"swab",
	"sweatshirt",
	"swimming trunks",
	"swing",
	"switch",
	"syringe",
	"table lamp",
	"tank",
	"tape player",
	"teapot",
	"teddy",
	"television",
	"tennis ball",
	"thatch",
	"theater curtain",
	"thimble",
	"thresher",
	"throne",
	"tile roof",
	"toaster",
	"tobacco shop",
	"toilet seat",
	"torch",
	"totem pole",
	"tow truck",
	"toyshop",
	"tractor",
	"trailer truck",
	"tray",
	"trench coat",
	"tricycle",
	"trimaran",
	"tripod",
	"triumphal arch",
	"trolleybus",
	"trombone",
	"tub",
	"turnstile",
	"typewriter keyboard",
	"umbrella",
	"unicycle",
	"upright",
	"vacuum",
	"vase",
	"vault",
	"velvet",
	"vending machine",
	"vestment",
	"viaduct",
	"violin",
	"volleyball",
	"waffle iron",
	"wall clock",
	"wallet",
	"wardrobe",
	"warplane",
	"washbasin",
	"washer",
	"water bottle",
	"water jug",
	"water tower",
	"whiskey jug",
	"whistle",
	"wig",
	"window screen",
	"window shade",
	"Windsor tie",
	"wine bottle",
	"wing",
	"wok",
	"wooden spoon",
	"wool",
	"worm fence",
	"wreck",
	"yawl",
	"yurt",
	"web site",
	"comic book",
	"crossword puzzle",
	"street sign",
	"traffic light",
	"book jacket",
	"menu",
	"plate",
	"guacamole",
	"consomme",
	"hot pot",
	"trifle",
	"ice cream",
	"ice lolly",
	"French loaf",
	"bagel",
	"pretzel",
	"cheeseburger",
	"hotdog",
	"mashed potato",
	"head cabbage",
	"broccoli",
	"cauliflower",
	"zucchini",
	"spaghetti squash",
	"acorn squash",
	"butternut squash",
	"cucumber",
	"artichoke",
	"bell pepper",
	"cardoon",
	"mushroom",
	"Granny Smith",
	"strawberry",
	"orange",
	"lemon",
	"fig",
	"pineapple",
	"banana",
	"jackfruit",
	"custard apple",
	"pomegranate",
	"hay",
	"carbonara",
	"chocolate sauce",
	"dough",
	"meat loaf",
	"pizza",
	"potpie",
	"burrito",
	"red wine",
	"espresso",
	"cup",
	"eggnog",
	"alp",
	"bubble",
	"cliff",
	"coral reef",
	"geyser",
	"lakeside",
	"promontory",
	"sandbar",
	"seashore",
	"valley",
	"volcano",
	"ballplayer",
	"groom",
	"scuba diver",
	"rapeseed",
	"daisy",
	"yellow lady's slipper",
	"corn",
	"acorn",
	"hip",
	"buckeye",
	"coral fungus",
	"agaric",
	"gyromitra",
	"stinkhorn",
	"earthstar",
	"hen-of-the-woods",
	"bolete",
	"ear",
	"toilet tissue"
};
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


#if TIMING
static int gettimediff_us2(struct timeval start, struct timeval end) {
	int sec = end.tv_sec - start.tv_sec;
	int usec = end.tv_usec - start.tv_usec;
	return sec * 1000000 + usec;
}
#endif
char *coco_classes[80] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

char *voc_classes[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant","sheep","sofa", "train", "tv/monitor"};

void print_json(model_t* model,vbx_cnn_io_ptr_t* io_buffers, int use_int8){
	
	FILE *fp;
	fp = fopen ("io.json", "w");
	
	//Take in model and iobuffers;
	fprintf(fp,"{\n");
	fprintf(fp,"\"inputs\":[\n");
	for(int i =0; i < (int)model_get_num_inputs(model);i++){
			if (i == 0) fprintf(fp,"{");
			int *in_dims = model_get_input_shape(model,i);
			int8_t* i8=(int8_t*)(uintptr_t)io_buffers[i];
			fix16_t* if16=(fix16_t*)(uintptr_t)io_buffers[i];

			fprintf(fp,"\"zero\":%d,", model_get_input_zeropoint(model,i)); 
			fprintf(fp,"\"scale\":%d,", (fix16_t)model_get_input_scale_fix16_value(model,i));
			fprintf(fp,"\"shape\":[");
			for(int j=0;j < (int)model_get_input_dims(model, i);j++){
				if(j == (int)model_get_input_dims(model, i)-1){
					fprintf(fp,"%d],\n",in_dims[j]);
				} else fprintf(fp,"%d,",in_dims[j]);
			}
			fprintf(fp,"\"data\":[ ");
			for (int j=0; j< (int)model_get_input_length(model, i); j++){
				if(j == (int)model_get_input_length(model, i)-1){
					if (use_int8) {
						fprintf(fp,"%d]\n", i8[j]);
					} else {
						fprintf(fp,"%d]\n", if16[j]);
					}
				} else {
					if (use_int8) {
						fprintf(fp,"%d,",i8[j]);
					} else {
						fprintf(fp,"%d,",if16[j]);
					}
				}
			}
			if(i == (int)model_get_num_inputs(model)-1){
				fprintf(fp,"}],\n");
			} else {
			        fprintf(fp,"},\n{");
			}
	}
	fprintf(fp,"\n\"outputs\":[\n");
	for (int o =0; o < (int)model_get_num_outputs(model); o++){
			if (o == 0) fprintf(fp,"{");
			int *out_dims = model_get_output_shape(model,o);

			int8_t *o8=(int8_t*)(uintptr_t)io_buffers[o+model_get_num_inputs(model)];
			uint8_t *o8u=(uint8_t*)(uintptr_t)io_buffers[o+model_get_num_inputs(model)];
			int16_t *o16=(int16_t*)(uintptr_t)io_buffers[o+model_get_num_inputs(model)];
			int32_t *o32=(int32_t*)(uintptr_t)io_buffers[o+model_get_num_inputs(model)];
			fix16_t* of16=(fix16_t*)(uintptr_t)io_buffers[o+model_get_num_inputs(model)];

			fprintf(fp,"\"zero\":%d,", model_get_output_zeropoint(model,o)); 
			fprintf(fp,"\"scale\":%d,", (fix16_t)model_get_output_scale_fix16_value(model,o));
			fprintf(fp,"\"shape\":[");
			for(int i=0;i < (int)model_get_output_dims(model, o);i++){
				if(i == (int)model_get_output_dims(model, o)-1){
					fprintf(fp,"%d],\n",out_dims[i]);
				} else fprintf(fp,"%d,",out_dims[i]);
			}
			fprintf(fp,"\"data\":[ ");
			for (int i=0; i< (int)model_get_output_length(model, o); i++){
				if(i == (int)model_get_output_length(model, o)-1){
					if (use_int8) {
						if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT8) {
							fprintf(fp,"%d]\n", o8[i]);
						} else if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_UINT8) {
							fprintf(fp,"%d]\n", o8u[i]);
						} else if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT16) {
							fprintf(fp,"%d]\n", o16[i]);
						} else if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT32) {
							fprintf(fp,"%d]\n", o32[i]);
						}
					} else {
						fprintf(fp,"%d]\n",of16[i]);
					}
				} else {
					if (use_int8) {
						if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT8) {
							fprintf(fp,"%d,", o8[i]);
						} else if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_UINT8) {
							fprintf(fp,"%d,", o8u[i]);
						} else if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT16) {
							fprintf(fp,"%d,", o16[i]);
						} else if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT32) {
							fprintf(fp,"%d,", o32[i]);
						}
					} else {
						fprintf(fp,"%d,",of16[i]);
					}
				}
			}
			if(o == (int)model_get_num_outputs(model)-1){
				fprintf(fp,"}]\n");
			} else {
			        fprintf(fp,"},\n{");
			}
	}
	fprintf(fp,"}\n");
	fclose(fp);
}

void preprocess_inputs(uint8_t* input, fix16_t scale, int32_t zero_point, int input_length, int int8_flag){
	fix16_t adjusted_scale = fix16_mul(scale,F16(255.0));
	fix16_t inv_adj_scale = fix16_div(F16(1.0),adjusted_scale);
	for(int c=0; c<input_length;c++){
		if (scale == 256){
			if(int8_flag)
				input[c] = (int8_t)((int32_t)input[c] - zero_point);
			else
				input[c] = (uint8_t)((int32_t)input[c] - zero_point);

		}
		else{
			if (int8_flag)
				input[c] = (int8_t)(fix16_mul((int32_t)input[c],inv_adj_scale) - zero_point);
			else
				input[c] = (uint8_t)(fix16_mul((int32_t)input[c],inv_adj_scale) - zero_point);
	
		}
	}
}
uint32_t fletcher32(const uint16_t *data, size_t len)
{
	uint32_t c0, c1;
	unsigned int i;

	for (c0 = c1 = 0; len >= 360; len -= 360) {
		for (i = 0; i < 360; ++i) {
			c0 = c0 + *data++;
			c1 = c1 + c0;
		}
		c0 = c0 % 65535;
		c1 = c1 % 65535;
	}
	for (i = 0; i < len; ++i) {
		c0 = c0 + *data++;
		c1 = c1 + c0;
	}
	c0 = c0 % 65535;
	c1 = c1 % 65535;
	return (c1 << 16 | c0);
}

fix16_t calcIou_LTRB(fix16_t* A, fix16_t* B){
    // pointers to elements (left, top, right, bottom)
    fix16_t left = MAX(A[0], B[0])>>4;  // adjust precision to prevent overflow; this is sufficient for 1920x1080
    fix16_t top = MAX(A[1], B[1])>>4;
    fix16_t right = MIN(A[2], B[2])>>4;
    fix16_t bottom = MIN(A[3], B[3])>>4;
    fix16_t i = fix16_mul(MAX(0,right-left), MAX(0,bottom-top));    // intersection
    fix16_t u = fix16_mul((A[2]-A[0])>>4, (A[3]-A[1])>>4) + fix16_mul((B[2]-B[0])>>4, (B[3]-B[1])>>4) - i;  // union
    if(u>0){
        fix16_t iou = fix16_div(i, u);
        return iou;
    }
    else{
        return 0;
    }
}
fix16_t calcIou_XYWH(fix16_t* A, fix16_t* B){
    // pointers to elements (x, y, width, height)
    fix16_t left = MAX(A[0] - (A[2]>>1), B[0] - (B[2]>>1));
    fix16_t right = MIN(A[0] + (A[2]>>1), B[0] + (B[2]>>1));
    fix16_t top = MAX(A[1] - (A[3]>>1), B[1] - (B[3]>>1));
    fix16_t bottom = MIN(A[1] + (A[3]>>1), B[1] + (B[3]>>1));
    fix16_t i = fix16_mul(MAX(0,right-left), MAX(0,bottom-top));    // intersection
    fix16_t u = fix16_mul(A[2], A[3]) + fix16_mul(B[2], B[3]) - i;  // union
    if(u>0)
        return fix16_div(i, u);
    else
        return 0;
}

static fix16_t fix16_eight = F16(8);
static fix16_t fix16_neight = F16(-8);
static fix16_t fix16_two = F16(2);
static fix16_t fix16_nhalf = F16(-0.5);
static fix16_t fix16_half = F16(0.5);

int partition_int8(int8_t* arr, int16_t *index, const int lo, const int hi)
{
	int8_t temp, pivot = arr[hi];
	int itemp;
	int j, i = lo - 1;

	for (j = lo; j < hi; j++) {
		//since many elements are the same as pivot, "randomly" put them on either side of pivot
		//otherwise very slow
		int cmp = arr[j] == pivot ? j&1: arr[j] < pivot;
		if (cmp) {
			i++;
			temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;

			itemp = index[j];
			index[j] = index[i];
			index[i] = itemp;
		}
	}

	if (arr[hi] < arr[i+1]) {
		temp = arr[hi];
		arr[hi] = arr[i+1];
		arr[i+1] = temp;

		itemp = index[hi];
		index[hi] = index[i+1];
		index[i+1] = itemp;
	}

	return i + 1;
}



void _quicksort_int8(int8_t *arr, int16_t *index, const int lo, const int hi)
{
	int split;
	if (lo < hi) {
		split = partition_int8(arr, index, lo, hi);
		_quicksort_int8(arr, index, lo, split-1);
		_quicksort_int8(arr, index, split+1, hi);
	}
}

void quicksort_int8(int8_t *arr, int16_t *index, const int length)
{
	_quicksort_int8(arr, index, 0, length-1);
}

int partition(fix16_t* arr, int16_t *index, const int lo, const int hi)
{
	fix16_t temp, pivot = arr[hi];
	int itemp;
	int j, i = lo - 1;

	for (j = lo; j < hi; j++) {
		//since many elements are the same as pivot, "randomly" put them on either side of pivot
		//otherwise very slow
		int cmp = arr[j] == pivot ? j&1: arr[j] < pivot;
		if (cmp) {
			i++;
			temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;

			itemp = index[j];
			index[j] = index[i];
			index[i] = itemp;
		}
	}

	if (arr[hi] < arr[i+1]) {
		temp = arr[hi];
		arr[hi] = arr[i+1];
		arr[i+1] = temp;

		itemp = index[hi];
		index[hi] = index[i+1];
		index[i+1] = itemp;
	}

	return i + 1;
}

void _quicksort(fix16_t *arr, int16_t *index, const int lo, const int hi)
{
	int split;
	if (lo < hi) {
		split = partition(arr, index, lo, hi);
		_quicksort(arr, index, lo, split-1);
		_quicksort(arr, index, split+1, hi);
	}
}

void quicksort(fix16_t *arr, int16_t *index, const int length)
{
	_quicksort(arr, index, 0, length-1);
}

void reverse(fix16_t* output_buffer[], int len){
	int left, right;
	for(left = 0, right = len-1; left < right; left++, right--){
		fix16_t* temp = output_buffer[left];
		output_buffer[left] = output_buffer[right];
		output_buffer[right] = temp;
	}
}

void int8_to_fix16(fix16_t* output, int8_t* input, int size, fix16_t f16_scale, int32_t zero_point){
	for (int i = 0; i < size; i++) {
		output[i] = fix16_mul(fix16_from_int((int32_t)(input[i]) - zero_point),f16_scale);
	}
}

fix16_t int8_to_fix16_single(int8_t input,fix16_t scale, int32_t zero_point){
	return fix16_mul(fix16_from_int((int32_t)(input) - zero_point),scale);
}

int8_t fix16_to_int8(fix16_t input, fix16_t f16_scale, int32_t zero_point){
	return (int8_t)(fix16_to_int(fix16_div(input,f16_scale)) +zero_point);
}

void post_classifier(fix16_t *outputs, const int out_sz, int16_t* output_index, int topk)
{
	int i;
	int16_t *idx = (int16_t*)malloc(out_sz*sizeof(int16_t));
	for (i = 0; i < out_sz; i++) idx[i] = i;
	quicksort(outputs, idx, out_sz);

	for(int i=0;i<topk;++i){
		output_index[i] = idx[out_sz-1 -i];
	}
	free(idx);
}

void post_classifier_int8(int8_t *outputs, const int out_sz, int16_t* output_index, int topk)
{
	int i;
	int16_t *idx = (int16_t*)malloc(out_sz*sizeof(int16_t));
	for (i = 0; i < out_sz; i++) idx[i] = i;
	quicksort_int8(outputs, idx, out_sz);

	for(int i=0;i<topk;++i){
		output_index[i] = idx[out_sz-1 -i];
	}
	free(idx);
}

int close(float a, float b, float threshold) {
	float diff = a - b;
	if (diff < 0) diff = diff * -1;
	if (diff > threshold) return 0;
	return 1;
}

static fix16_t exp_lut[] ={0x00010000,0x00011082,0x00012216,0x000134cb,0x000148b5,0x00015de9,0x0001747a,0x00018c80,
	0x0001a612,0x0001c14b,0x0001de45,0x0001fd1d,0x00021df3,0x000240e7,0x0002661c,0x00028db8,
	0x0002b7e1,0x0002e4c2,0x00031489,0x00034764,0x00037d87,0x0003b727,0x0003f47f,0x000435cc,
	0x00047b4f,0x0004c54e,0x00051413,0x000567ec,0x0005c12d,0x00062030,0x00068554,0x0006f0fe,
	0x00076399,0x0007dd98,0x00085f76,0x0008e9b4,0x00097cdc,0x000a1982,0x000ac042,0x000b71c3,
	0x000c2eb7,0x000cf7db,0x000dcdf8,0x000eb1e4,0x000fa483,0x0010a6c8,0x0011b9b5,0x0012de5d,
	0x001415e5,0x00156185,0x0016c288,0x00183a4f,0x0019ca53,0x001b7423,0x001d396a,0x001f1bed,
	0x00211d8e,0x0023404f,0x00258654,0x0027f1e2,0x002a8565,0x002d4371,0x00302ec5,0x00334a4b,
	0x00369920,0x003a1e92,0x003dde28,0x0041dba1,0x00461afc,0x004aa077,0x004f7099,0x00549032,
	0x005a0462,0x005fd29e,0x006600b5,0x006c94d5,0x00739593,0x007b09f0,0x0082f962,0x008b6bd7,
	0x009469c4,0x009dfc28,0x00a82c94,0x00b3053b,0x00be90f6,0x00cadb53,0x00d7f09b,0x00e5dde6,
	0x00f4b122,0x01047924,0x011545b4,0x012727a1,0x013a30cf,0x014e7447,0x01640650,0x017afc7c,
	0x01936dc5,0x01ad729d,0x01c9250b,0x01e6a0c5,0x02060348,0x02276bf9,0x024afc45,0x0270d7bd,
	0x02992442,0x02c40a22,0x02f1b447,0x0322505f,0x03560f0b,0x038d240c,0x03c7c67e,0x04063107,
	0x0448a216,0x048f5c23,0x04daa5ee,0x052acac6,0x05801ad7,0x05daeb78,0x063b9782,0x06a27fa8,
	0x07100adb,0x0784a6b0,0x0800c7cc,0x0884ea5b,0x09119289,0x09a74d0c,0x0a46afaa,0x0af059d2,
	0x00000015,0x00000017,0x00000018,0x0000001a,0x0000001c,0x0000001e,0x0000001f,0x00000022,
	0x00000024,0x00000026,0x00000029,0x0000002b,0x0000002e,0x00000031,0x00000034,0x00000038,
	0x0000003b,0x0000003f,0x00000043,0x00000048,0x0000004c,0x00000051,0x00000056,0x0000005c,
	0x00000062,0x00000068,0x0000006f,0x00000076,0x0000007e,0x00000086,0x0000008f,0x00000098,
	0x000000a2,0x000000ac,0x000000b8,0x000000c3,0x000000d0,0x000000de,0x000000ec,0x000000fb,
	0x0000010b,0x0000011d,0x0000012f,0x00000143,0x00000157,0x0000016e,0x00000185,0x0000019e,
	0x000001b9,0x000001d6,0x000001f4,0x00000214,0x00000236,0x0000025b,0x00000282,0x000002ab,
	0x000002d8,0x00000306,0x00000338,0x0000036e,0x000003a6,0x000003e3,0x00000423,0x00000467,
	0x000004b0,0x000004fd,0x00000550,0x000005a7,0x00000605,0x00000668,0x000006d2,0x00000743,
	0x000007bb,0x0000083a,0x000008c2,0x00000953,0x000009ed,0x00000a90,0x00000b3f,0x00000bf9,
	0x00000cbe,0x00000d91,0x00000e71,0x00000f5f,0x0000105d,0x0000116b,0x0000128b,0x000013bd,
	0x00001503,0x0000165e,0x000017cf,0x00001958,0x00001afb,0x00001cb8,0x00001e93,0x0000208b,
	0x000022a5,0x000024e1,0x00002742,0x000029ca,0x00002c7c,0x00002f5a,0x00003268,0x000035a9,
	0x0000391f,0x00003cce,0x000040ba,0x000044e6,0x00004958,0x00004e13,0x0000531c,0x00005878,
	0x00005e2d,0x00006440,0x00006ab7,0x00007199,0x000078ed,0x000080b9,0x00008906,0x000091dd,
	0x00009b45,0x0000a549,0x0000aff2,0x0000bb4b,0x0000c75f,0x0000d43b,0x0000e1eb,0x0000f07d};

fix16_t fix16_exp_lut(fix16_t in){
	if (in < fix16_neight) return 0x15;
	if (in >= fix16_eight) return 0xaf059d0;

	uint8_t index = in >> 12;
	return exp_lut[index];
}

#define fix16_exp fix16_exp_lut
fix16_t fix16_logistic_activate(fix16_t x){ return fix16_div(fix16_one, fix16_add(fix16_one, fix16_exp(-x)));} // 1 div, 1 exp

void fix16_softmax(fix16_t *input, int n, fix16_t *output)
{
	fix16_t e, sum = 0;
	fix16_t largest = fix16_minimum;
	for(int i = 0; i < n; ++i){
		if(input[i] > largest) largest = input[i];
	}
	for(int i = 0; i < n; ++i){
		e = fix16_exp(fix16_sub(input[i], largest));
		sum = fix16_add(sum, e);
		output[i] = e;
	}
	fix16_t inv_sum = fix16_div(fix16_one, sum);
	for(int i = 0; i < n; ++i){
		output[i] = fix16_mul(output[i], inv_sum);
	}
}

int fix16_box_iou(fix16_box box_1, fix16_box box_2, fix16_t thresh)
{
	//return true if the IOU score of box_1 and box2 > threshold
	int width_of_overlap_area = ((box_1.xmax < box_2.xmax) ? box_1.xmax : box_2.xmax) - ((box_1.xmin > box_2.xmin) ? box_1.xmin : box_2.xmin);
	int height_of_overlap_area = ((box_1.ymax < box_2.ymax) ? box_1.ymax : box_2.ymax) - ((box_1.ymin > box_2.ymin) ? box_1.ymin : box_2.ymin);

	int area_of_overlap;
	if (width_of_overlap_area < 0 || height_of_overlap_area < 0) {
		return 0;
	} else {
		area_of_overlap = width_of_overlap_area * height_of_overlap_area;
		if( area_of_overlap <0){
			//overflow,divide boxes by 4 and try again
			box_1.xmin >>=2;
			box_1.ymin >>=2;
			box_1.xmax >>=2;
			box_1.ymax >>=2;
			box_2.xmin >>=2;
			box_2.ymin >>=2;
			box_2.xmax >>=2;
			box_2.ymax >>=2;
			return fix16_box_iou(box_1,box_2,thresh);
		}
	}

	int box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
	int box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
	int area_of_union = box_1_area + box_2_area - area_of_overlap;
	if (area_of_union == 0) return 0;

	return area_of_overlap > fix16_mul(thresh, area_of_union);
}

//merge boxes into first box, see python merge_inner_detection_object_pair
void merge_boxes(fix16_box *box_1, poses_t *pose_1, fix16_box *box_2, poses_t *pose_2)
{

	fix16_t box_1_area = (fix16_t)((box_1->ymax - box_1->ymin) * (box_1->xmax - box_1->xmin));
	fix16_t box_2_area = (fix16_t)((box_2->ymax - box_2->ymin) * (box_2->xmax - box_2->xmin));
	fix16_t inv_area = fix16_div(F16(1.), fix16_add(box_1_area, box_2_area));

	// merged confidence
        box_1->confidence = fix16_mul(inv_area, fix16_add(fix16_mul(box_1_area, box_1->confidence), 
					fix16_mul(box_2_area, box_2->confidence))); 

	// merged keypoints
	for(int p = 0; p <17;p++){
		pose_1->scores[p] = fix16_mul(inv_area, fix16_add(fix16_mul(box_1_area, pose_1->scores[p]), 
						fix16_mul(box_2_area, pose_2->scores[p]))); 
		pose_1->keypoints[p][0] = fix16_mul(inv_area, fix16_add(fix16_mul(box_1_area, pose_1->keypoints[p][0]), 
						fix16_mul(box_2_area, pose_2->keypoints[p][0]))); 
		pose_1->keypoints[p][1] = fix16_mul(inv_area, fix16_add(fix16_mul(box_1_area, pose_1->keypoints[p][1]), 
						fix16_mul(box_2_area, pose_2->keypoints[p][1]))); 
	}

	box_1->ymin = MIN(box_1->ymin, box_2->ymin);
	box_1->xmin = MIN(box_1->xmin, box_2->xmin);
	box_1->ymax = MAX(box_1->ymax, box_2->ymax);
	box_1->xmax = MAX(box_1->xmax, box_2->xmax);

	return;
}


void fix16_do_nmm(fix16_box *boxes, poses_t *poses, int total, fix16_t iou_thresh)
{
	int merge_groups[20][200];
	int merge_groups_count[20] = {0.};
	int merge_group_idx = 0;
	int idx = 0, midx, remaining;

	int unmerged[200];
	for(int i = 0; i < total; i++){
		unmerged[i] = 1;
	}

	remaining = total;
	while(remaining) { // create merge groups
		for(int i = 0; i < total; i++){
			if(unmerged[total-1-i]) {
				idx = total-1-i;
				merge_groups[merge_group_idx][0] = idx;
				merge_groups_count[merge_group_idx]++;
				unmerged[idx] = 0;
				remaining--;
				break;
			}
		}
		if (remaining == 0) break;

		for(int i = 0; i < total; i++){
			if (unmerged[total-1-i]) {
				if (fix16_box_iou(boxes[total-1-i], boxes[idx], iou_thresh)){
					midx = merge_groups_count[merge_group_idx];
					merge_groups[merge_group_idx][midx] = total-1-i;
					merge_groups_count[merge_group_idx]++;
					unmerged[total-1-i] = 0;
					remaining--;
				}
			}
		}
		merge_group_idx++;
	}

	fix16_t thresh = F16(0.5);
	for(int g = 0; g < 20; g++){
		if (merge_groups_count[g]) {
			// merge group into first box
			for(int m = 1; m < merge_groups_count[g]; m++){
				if (fix16_box_iou(boxes[merge_groups[g][0]], boxes[merge_groups[g][m]], thresh)){
					merge_boxes(boxes+merge_groups[g][0],
						poses+merge_groups[g][0],
						boxes+merge_groups[g][m],
						poses+merge_groups[g][m]);
					break;
				} else {
					break;
				}
			}
			//zero out all but first box
			for(int m = 1; m < merge_groups_count[g]; m++){
				boxes[merge_groups[g][m]].confidence = 0;
			}
		}
	}

	for(int i = 0; i < total; i++){
		if (boxes[i].confidence == 0) continue;
		for(int j = i+1; j < total; j++){
			if (fix16_box_iou(boxes[i], boxes[j], iou_thresh)){
				boxes[j].confidence = 0;
			}
		}
	}
}




void fix16_do_nms(fix16_box *boxes, int total, fix16_t iou_thresh)
{
	for(int i = 0; i < total; i++){
		if (boxes[i].confidence == 0) continue;
		for(int j = i+1; j < total; j++){
			if (fix16_box_iou(boxes[i], boxes[j], iou_thresh)){
				boxes[j].confidence = 0;
			}
		}
	}
}

void fix16_sort_boxes(fix16_box *boxes, poses_t *poses, int total)
{
	for (int i = 0; i < total-1; i++) {
		int max_id = i;
		for(int j = i+1; j < total; j++){
			if (boxes[max_id].confidence < boxes[j].confidence) {
				max_id = j;
			}
		}

		if (max_id != i) {
			//swap max_id with i
			fix16_box tmp = boxes[i];
			memcpy((void*)(boxes+i), (void*)(boxes+max_id), sizeof(fix16_box));
			memcpy((void*)(boxes+max_id), &tmp, sizeof(fix16_box));
			if (poses != NULL) {
				poses_t tmp = poses[i];
				memcpy((void*)(poses+i), (void*)(poses+max_id), sizeof(poses_t));
				memcpy((void*)(poses+max_id), &tmp, sizeof(poses_t));
			}
		}
	}
}


int fix16_clean_boxes(fix16_box *boxes, poses_t *poses, int total, int width, int height)
{
	int b=0;
	for(int i = 0; i < total; i++){
		if (boxes[i].confidence > 0) {
			fix16_box box = boxes[i];
			box.xmin = box.xmin < 0 ? 0 : box.xmin;
			box.xmax = box.xmax >width ? width : box.xmax;
			box.ymin = box.ymin < 0 ? 0 : box.ymin;
			box.ymax = box.ymax >height ? height : box.ymax;
			boxes[b] = box;
			if (poses != NULL) {
				poses_t tmp = poses[i];
				memcpy((void*)(poses+b), &tmp, sizeof(poses_t));
			}
			b++;
		}
	}
	return b;
}

//////////
int fix16_get_region_boxes_int8(int8_t *predictions, int zero_point, fix16_t scale_out, fix16_t *biases, const int w, const int h, const int ln, 
		const int classes, fix16_t w_ratio, fix16_t h_ratio, fix16_t thresh, fix16_t log_odds, fix16_box *boxes,
		int max_boxes, const int do_logistic, const int do_softmax, const int version)
{
	int box_count = 0;

	int num_size=(classes+5) *w*h;
	fix16_t box[classes+4];
	for(int r=0;r<h;r++){
		for(int c=0;c<w;c++){
			fix16_t row = fix16_from_int(r);
			fix16_t col = fix16_from_int(c);
			for(int n = 0; n < ln; ++n){
				int p_index = n*num_size+4*w*h+r*w+c;
				fix16_t scale = fix16_smul(fix16_from_int((int8_t)predictions[p_index] - zero_point), scale_out);
				if (do_logistic) {
					if (scale < log_odds) continue;
					scale = fix16_logistic_activate(scale);
				}
				if (scale < thresh) continue;

				const int class_offset = 4;
				for(int j=0; j < class_offset;++j){
					box[j] = fix16_smul(fix16_from_int((int8_t)predictions[n*num_size+j*w*h+r*w+c] - zero_point), scale_out);
				}
				for(int j=0;j<classes;++j){
					box[j+class_offset] = fix16_smul(fix16_from_int((int8_t)predictions[n*num_size+(j+class_offset+1)*w*h+r*w+c] - zero_point), scale_out);
				}

				fix16_t bx, by, bw, bh;
				if (version > 3) {
					// (col+logisitic(box)*2-0.5) * ratio
					bx = fix16_mul(fix16_add(fix16_add(col, fix16_mul(fix16_logistic_activate(box[0]), fix16_two)), fix16_nhalf), w_ratio);
					by = fix16_mul(fix16_add(fix16_add(row, fix16_mul(fix16_logistic_activate(box[1]), fix16_two)), fix16_nhalf), h_ratio);

					bh = fix16_mul(fix16_logistic_activate(box[3]), fix16_two);
					bh = fix16_mul(fix16_mul(bh, bh), biases[2*n+1]);

					// (logisitic(box)*2)**2 * anchor
					bw = fix16_mul(fix16_logistic_activate(box[2]), fix16_two);
					bw = fix16_mul(fix16_mul(bw, bw), biases[2*n]);
				} else {
					bx = fix16_mul(fix16_add(col, fix16_logistic_activate(box[ 0])), w_ratio);
					by = fix16_mul(fix16_add(row, fix16_logistic_activate(box[ 1])), h_ratio);
					bw = fix16_mul(fix16_exp(box[2]), biases[2*n]);
					bh = fix16_mul(fix16_exp(box[3]), biases[2*n+1]);
				}

				if (do_softmax) {
					if (version < 3) {
						fix16_softmax(box + class_offset, classes, box + class_offset);
					} else {
						for(int j=0;j<classes;++j){
							box[j+class_offset] = fix16_logistic_activate(box[j+class_offset]);
						}
					}
				}

				for(int j = 0; j < classes; j++){
					fix16_t prob = fix16_mul(scale, box[class_offset+j]);
					if (prob > thresh) {
						fix16_t xmin = fix16_sub(bx, bw >> 1);
						fix16_t ymin = fix16_sub(by, bh >> 1);
						fix16_t xmax = fix16_add(xmin, bw);
						fix16_t ymax = fix16_add(ymin, bh);

						boxes[box_count].xmin = fix16_to_int(xmin);
						boxes[box_count].ymin = fix16_to_int(ymin);
						boxes[box_count].xmax = fix16_to_int(xmax);
						boxes[box_count].ymax = fix16_to_int(ymax);
						boxes[box_count].confidence = prob;
						boxes[box_count].class_id = j;
						box_count++;
						if(box_count == max_boxes){
							return box_count;
						}
					}
				}
			}
		}
	}
	return box_count;
}

int post_process_yolo_int8(int8_t **outputs, const int num_outputs, int zero_points[], fix16_t scale_outs[], 
	 yolo_info_t *cfg, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes)
{
	int total_box_count = 0;
	int input_h  = cfg[0].input_dims[1];
	int input_w  = cfg[0].input_dims[2];
	fix16_t *anchors = cfg[0].anchors;

	fix16_t fix16_log_odds = fix16_log(fix16_div(thresh, fix16_sub(fix16_one, thresh)));
	for (int o = 0; o < num_outputs; o++) {
		int8_t *out8 = outputs[o];

		int num_per_output = cfg[o].num;
		if (num_outputs > 1) num_per_output = cfg[o].mask_length;

		fix16_t fix16_biases[2*num_per_output];

		int h  = cfg[o].output_dims[1];
		int w  = cfg[o].output_dims[2];
		fix16_t h_ratio = fix16_from_int(input_h/h);
		fix16_t w_ratio = fix16_from_int(input_w/w);

		for (int i = 0; i < num_per_output; i++) {
			int mask = i;
			if (num_outputs > 1) mask = cfg[o].mask[i];

			if (cfg[o].version == 2) {
				fix16_biases[2*i] = fix16_mul(anchors[2*mask], w_ratio);
				fix16_biases[2*i+1] = fix16_mul(anchors[2*mask+1], h_ratio);
			} else {
				fix16_biases[2*i] = anchors[2*mask];
				fix16_biases[2*i+1] = anchors[2*mask+1];
			}
		}

		int fix16_box_count = fix16_get_region_boxes_int8(out8, zero_points[o], scale_outs[o], fix16_biases, w, h, num_per_output, cfg[o].classes, w_ratio,
				h_ratio, thresh, fix16_log_odds,
				fix16_boxes + total_box_count,max_boxes-total_box_count, 1, 1, cfg[o].version);
		fflush(stdout);

		// copy boxes
		total_box_count += fix16_box_count;
		if(total_box_count == max_boxes){
			break;
		}
	}
	fix16_sort_boxes(fix16_boxes, NULL, total_box_count);
	fix16_do_nms(fix16_boxes, total_box_count, overlap);
	int clean_box_count = fix16_clean_boxes(fix16_boxes, NULL, total_box_count, input_w, input_h);

	return clean_box_count;
}
//////

int fix16_get_region_boxes(fix16_t *predictions, fix16_t *biases, const int w, const int h, const int ln, const int classes, fix16_t w_ratio, fix16_t h_ratio, fix16_t thresh, fix16_t log_odds, fix16_box *boxes,int max_boxes, const int do_logistic, const int do_softmax, const int version)
{
	int box_count = 0;

	int num_size=(classes+5) *w*h;
	fix16_t box[classes+4];
	for(int r=0;r<h;r++){
		for(int c=0;c<w;c++){
			fix16_t row = fix16_from_int(r);
			fix16_t col = fix16_from_int(c);
			for(int n = 0; n < ln; ++n){
				int p_index = n*num_size+4*w*h+r*w+c;
				fix16_t scale = predictions[p_index];
				if (do_logistic) {
					if (scale < log_odds) continue;
					scale = fix16_logistic_activate(scale);
				}
				if (scale < thresh) continue;

				const int class_offset = 4;
				for(int j=0; j < class_offset;++j){
					box[j] = predictions[n*num_size+j*w*h+r*w+c];
				}
				for(int j=0;j<classes;++j){
					box[j+class_offset] = predictions[n*num_size+(j+class_offset+1)*w*h+r*w+c];
				}

				fix16_t bx, by, bw, bh;
				if (version > 3) {
					// (col+logisitic(box)*2-0.5) * ratio
					bx = fix16_mul(fix16_add(fix16_add(col, fix16_mul(fix16_logistic_activate(box[0]), fix16_two)), fix16_nhalf), w_ratio);
					by = fix16_mul(fix16_add(fix16_add(row, fix16_mul(fix16_logistic_activate(box[1]), fix16_two)), fix16_nhalf), h_ratio);

					bh = fix16_mul(fix16_logistic_activate(box[3]), fix16_two);
					bh = fix16_mul(fix16_mul(bh, bh), biases[2*n+1]);

					// (logisitic(box)*2)**2 * anchor
					bw = fix16_mul(fix16_logistic_activate(box[2]), fix16_two);
					bw = fix16_mul(fix16_mul(bw, bw), biases[2*n]);
				} else {
					bx = fix16_mul(fix16_add(col, fix16_logistic_activate(box[ 0])), w_ratio);
					by = fix16_mul(fix16_add(row, fix16_logistic_activate(box[ 1])), h_ratio);
					bw = fix16_mul(fix16_exp(box[2]), biases[2*n]);
					bh = fix16_mul(fix16_exp(box[3]), biases[2*n+1]);
				}

				if (do_softmax) {
					if (version < 3) {
						fix16_softmax(box + class_offset, classes, box + class_offset);
					} else {
						for(int j=0;j<classes;++j){
							box[j+class_offset] = fix16_logistic_activate(box[j+class_offset]);
						}
					}
				}

				for(int j = 0; j < classes; j++){
					fix16_t prob = fix16_mul(scale, box[class_offset+j]);
					if (prob > thresh) {
						fix16_t xmin = fix16_sub(bx, bw >> 1);
						fix16_t ymin = fix16_sub(by, bh >> 1);
						fix16_t xmax = fix16_add(xmin, bw);
						fix16_t ymax = fix16_add(ymin, bh);

						boxes[box_count].xmin = fix16_to_int(xmin);
						boxes[box_count].ymin = fix16_to_int(ymin);
						boxes[box_count].xmax = fix16_to_int(xmax);
						boxes[box_count].ymax = fix16_to_int(ymax);
						boxes[box_count].confidence = prob;
						boxes[box_count].class_id = j;
						box_count++;
						if(box_count == max_boxes){
							return box_count;
						}
					}
				}
			}
		}
	}
	return box_count;
}


int post_process_ultra_nms_int8(int8_t *output, int output_boxes, int input_h, int input_w,fix16_t f16_scale, int32_t zero_point, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int boxes_len, const int num_classes)
{
#if TIMING	
static struct timeval tv1, tv2,tv0;
gettimeofday(&tv0, NULL); 	
#endif

#if TIMING
gettimeofday(&tv1, NULL); 
printf("copy over outputs: %d ms\n",(gettimediff_us2(tv0, tv1) / 1000));	
#endif
		int total_box_count = 0;
		int8_t thresh0 = fix16_to_int8(thresh,f16_scale,zero_point);
		for(int i =0; i< output_boxes; i++){		
			//fix16_t max_score = F16(-1.0);
			int8_t max_score = -128;
			int max_score_ind = 0;
					
			for(int c = 4; c <(4+num_classes);c++){
				if(output[i*(4+num_classes) +c]>  max_score){
					max_score=output[i*(4+num_classes) + c];
					max_score_ind = c-4;
				}
			}

			if(max_score > thresh0){
				fix16_boxes[total_box_count].confidence = int8_to_fix16_single(max_score,f16_scale,zero_point);
				fix16_boxes[total_box_count].class_id = max_score_ind;

				fix16_t x = int8_to_fix16_single(output[i*(4+num_classes)+0],f16_scale, zero_point);
				fix16_t y = int8_to_fix16_single(output[i*(4+num_classes)+1],f16_scale, zero_point);
				fix16_t w = int8_to_fix16_single(output[i*(4+num_classes)+2],f16_scale, zero_point);
				fix16_t h = int8_to_fix16_single(output[i*(4+num_classes)+3],f16_scale, zero_point);

				fix16_boxes[total_box_count].xmin = fix16_to_int((x - fix16_mul(w,fix16_half))*input_w);
				fix16_boxes[total_box_count].xmax = fix16_to_int((x + fix16_mul(w,fix16_half))*input_w);
				fix16_boxes[total_box_count].ymin = fix16_to_int((y - fix16_mul(h,fix16_half))*input_h);
				fix16_boxes[total_box_count].ymax = fix16_to_int((y + fix16_mul(h,fix16_half))*input_h);
				total_box_count++;

			}
			if(total_box_count>=boxes_len)
				break;
		}

		fix16_sort_boxes(fix16_boxes, NULL, total_box_count);

		fix16_do_nms(fix16_boxes, total_box_count, overlap);
		int clean_box_count = fix16_clean_boxes(fix16_boxes, NULL, total_box_count, input_w, input_h);
#if TIMING
gettimeofday(&tv2, NULL); 	
	
printf("total time: %d ms\n",(gettimediff_us2(tv0, tv2) / 1000));		
printf("memcpy: %d ms\n",(gettimediff_us2(tv0, tv1) / 1000));		
printf("processing: %d ms\n",(gettimediff_us2(tv1, tv2) / 1000));		
#endif
		return clean_box_count;


}

int ultralytics_process_box(fix16_t *xywh, fix16_t* arr, fix16_t angle, const int j, const int i, const int size)
{
	fix16_t soft[16];
	fix16_t in[16];
	fix16_t v[4];
	fix16_t scale = fix16_div(fix16_one, fix16_from_int(size));

	fix16_t anchor[] ={F16(0),F16(1),F16(2),F16(3),F16(4),F16(5),F16(6),F16(7),
		F16(8),F16(9),F16(10),F16(11),F16(12),F16(13),F16(14),F16(15)};
	for (int c = 0; c < 4; c++) {
		fix16_t sum = 0;
		for (int s = 0; s < 16; s++) {
			in[s] = arr[(c*16+s)*size*size+j*size+i];
		}
		fix16_softmax(in, 16, soft);
		for (int s = 0; s < 16; s++) {
			sum = fix16_add(sum, fix16_mul(anchor[s], soft[s]));
		}
		v[c] = fix16_mul(sum, scale);
	}

	fix16_t cx = fix16_add(fix16_mul(scale, F16(0.5)), fix16_mul(fix16_from_int(i),scale));//cx = 1/bh/2 + y/bh
	fix16_t cy = fix16_add(fix16_mul(scale, F16(0.5)), fix16_mul(fix16_from_int(j),scale));//cy = 1/bw/2 + x/bw
											       
	xywh[0] = fix16_mul(fix16_sub(v[2], v[0]), F16(0.5));
	xywh[1] = fix16_mul(fix16_sub(v[3], v[1]), F16(0.5));
	if (angle != fix16_minimum) {
	}
	xywh[0] = fix16_add(cx, xywh[0]);
	xywh[1] = fix16_add(cy, xywh[1]);
							 
	xywh[2] = fix16_add(v[2], v[0]);
	xywh[3] = fix16_add(v[3], v[1]);

	
	return 0;
}

int ultralytics_process_box_int8(fix16_t *xywh, int8_t* arr, fix16_t angle, const int h, const int w, const int H, const int W, int zero_point, fix16_t scale_output)
{
	fix16_t soft[16];
	fix16_t v[4];
	fix16_t anchor[] ={F16(0),F16(1),F16(2),F16(3),F16(4),F16(5),F16(6),F16(7),
		F16(8),F16(9),F16(10),F16(11),F16(12),F16(13),F16(14),F16(15)};
	fix16_t inv_H = fix16_div(fix16_one, fix16_from_int(H));
	fix16_t inv_W = fix16_div(fix16_one, fix16_from_int(W));

	for (int c = 0; c < 4; c++) {
		fix16_t sum = 0;
		for (int s = 0; s < 16; s++) {
			soft[s] = int8_to_fix16_single(arr[(c*16+s)*H*W+h*W+w],scale_output,zero_point);
		}
		fix16_softmax(soft, 16, soft);
		for (int s = 0; s < 16; s++) {
			sum = fix16_add(sum, fix16_mul(anchor[s], soft[s]));
		}
		if(c%2==0){
			v[c] = fix16_mul(sum,inv_W);
		}
		else{
			v[c] = fix16_mul(sum,inv_H);
		}
	}

	fix16_t cx = fix16_add(fix16_mul(inv_W, F16(0.5)), fix16_mul(fix16_from_int(w),inv_W));//cx = 1/bw/2 + y/bw
	fix16_t cy = fix16_add(fix16_mul(inv_H, F16(0.5)), fix16_mul(fix16_from_int(h),inv_H));//cy = 1/bh/2 + x/bh
	xywh[0] = fix16_mul(fix16_sub(v[2], v[0]), F16(0.5));
	xywh[1] = fix16_mul(fix16_sub(v[3], v[1]), F16(0.5));
	if (angle != fix16_minimum) {
	    //TODO
	    fix16_t x0 = xywh[0];
	    fix16_t y0 = xywh[1];
	    xywh[0] = fix16_sub(fix16_mul(fix16_cos(angle),x0), fix16_mul(fix16_sin(angle),y0));
	    xywh[1] = fix16_add(fix16_mul(fix16_sin(angle),x0), fix16_mul(fix16_cos(angle),y0));
	}
	xywh[0] = fix16_add(cx, xywh[0]);
	xywh[1] = fix16_add(cy, xywh[1]);
							 
	xywh[2] = fix16_add(v[2], v[0]);
	xywh[3] = fix16_add(v[3], v[1]);

	return 0;
}


int post_process_ultra_int8(int8_t **outputs, int* outputs_shape[], fix16_t *post, fix16_t thresh, int zero_points[], fix16_t scale_outs[], const int max_boxes, const int is_obb, const int is_pose,int num_outputs)
{
	int total_count = 0;
	int C = outputs_shape[0][1];	// number of classes (80 for COCO)
	fix16_t fix16_log_odds = fix16_log(fix16_div(thresh, fix16_sub(fix16_one, thresh)));
	bool has_argmax = false;
	if (num_outputs ==9)
		has_argmax = true;
	int outputs_per_stride = 2; //increment should be done by stride sets, (usually 3 sets)
	for(int o=0; o < 6; o+=outputs_per_stride){
		int H = outputs_shape[o][2];
		int W = outputs_shape[o][3];
		int8_t *out8 = outputs[o];
		int temp_zero = zero_points[o];
		fix16_t temp_scale = scale_outs[o];
		int8_t i8_log_odds = fix16_to_int8(fix16_log_odds,temp_scale,temp_zero);

		int valid_locations[H][W];
		for(int h=0; h<H; h++)
			for(int w=0; w<W; w++)
				valid_locations[h][w] = 0;
		if(has_argmax){
			uint8_t *argmax = outputs[(o/2+6)];
			for(int h=0; h<H; h++){
				for(int w=0; w<W; w++){
					uint8_t C = argmax[h*W+w];
					if(out8[C*H*W + h*W + w]>i8_log_odds){	// only process likely scores
						valid_locations[h][w] = 1;
					}
				}
			}
		}
		else{
			for(int c=0; c<C; c++){
				for(int h=0; h<H; h++){
					for(int w=0; w<W; w++){
						if(out8[ c*H*W + h*W + w]>i8_log_odds){	// only process likely scores
							valid_locations[h][w] = 1;
						}
					}
				}
			}
		}
		fix16_t inv_H = fix16_div(fix16_one, fix16_from_int(H));
		fix16_t inv_W = fix16_div(fix16_one, fix16_from_int(W));

		for(int h=0; h<H; h++){
			for(int w=0; w<W; w++){
				if(valid_locations[h][w] && total_count<max_boxes){
					fix16_t *xywh = post + total_count*(C+4+is_obb+!!is_pose*51);
					fix16_t angle = fix16_minimum;
					if (is_obb) {
						int8_t angle8 = outputs[6+o/2][h*W+w]; //assumed to be in order
						angle = fix16_logistic_activate(int8_to_fix16_single(angle8, scale_outs[6+o/2],zero_points[6+o/2]));
						angle = fix16_sub(angle, F16(0.25));
						angle = fix16_mul(angle, F16(3.141592741));

						post[total_count*(C+4+is_obb+!!is_pose*51)+4+C] = angle;
					}
					ultralytics_process_box_int8(xywh, outputs[o+1], angle, h, w, H, W, zero_points[o+1], scale_outs[o+1]);
					if (is_pose) {
						fix16_t py = -1;
						fix16_t px = -1;
						fix16_t score = -1;

						for(int p=0; p<17; p++){
							int idx_x = ((p*3+0)*H*W) + h*W+w;
							int idx_y = ((p*3+1)*H*W) + h*W+w;
							int idx_s = ((p*3+2)*H*W) + h*W+w;

							score = fix16_logistic_activate(int8_to_fix16_single(outputs[6+o/2][idx_s], scale_outs[6+o/2],zero_points[6+o/2]));
							px = fix16_mul(int8_to_fix16_single(outputs[6+o/2][idx_x], scale_outs[6+o/2],zero_points[6+o/2]), F16(2.));
							py = fix16_mul(int8_to_fix16_single(outputs[6+o/2][idx_y], scale_outs[6+o/2],zero_points[6+o/2]), F16(2.));
							if(is_pose==2){
								idx_x = ((p*2+0)*H*W) + h*W+w;
								idx_y = ((p*2+1)*H*W) + h*W+w;
								idx_s = ((p*1+0)*H*W) + h*W+w;

								score = fix16_logistic_activate(int8_to_fix16_single(outputs[6+o+1][idx_s], scale_outs[6+o],zero_points[6+o]));
								px = fix16_mul(int8_to_fix16_single(outputs[6+o][idx_x], scale_outs[6+o],zero_points[6+o]), F16(2.));
								py = fix16_mul(int8_to_fix16_single(outputs[6+o][idx_y], scale_outs[6+o],zero_points[6+o]), F16(2.));
							}
							px = fix16_add(px, fix16_from_int(w));
							py = fix16_add(py, fix16_from_int(h));

							px = fix16_mul(px, inv_W);
							py = fix16_mul(py, inv_H);
							
							post[total_count*(C+4+is_obb+!!is_pose*51)+4+C+3*p+0] = px;
							post[total_count*(C+4+is_obb+!!is_pose*51)+4+C+3*p+1] = py;
							post[total_count*(C+4+is_obb+!!is_pose*51)+4+C+3*p+2] = score;							
						}						
					}

					for(int c=0; c<C; c++){
						int8_t val = out8[c*H*W + h*W + w];
						if(val > i8_log_odds){
							post[total_count*(C+4+is_obb+!!is_pose*51)+4+c] = fix16_logistic_activate(int8_to_fix16_single(val,temp_scale,temp_zero));
						} else {
							post[total_count*(C+4+is_obb+!!is_pose*51)+4+c] = 0;
						}
					}
					total_count++;
				}
			}
		}
	}
	return total_count;

}


int post_process_ultra_nms(fix16_t *output, int output_boxes, int input_h, int input_w, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], poses_t poses[], int boxes_len, const int num_classes, const int is_obb, const int is_pose)
{
		int total_box_count = 0;

		int output_sz = 4+num_classes;
		if (is_obb) output_sz += 1;
		if (is_pose) output_sz += 17*3;

		for(int i=0; i<output_boxes; i++){
			fix16_t max_score = F16(-1.0);
			int max_score_ind = 0;
			for(int c = 4; c <4+num_classes;c++){
				if(output[i*output_sz +c]>  max_score){
					max_score=output[i*output_sz + c];
					max_score_ind = c-4;
				}
			}
			if(max_score > thresh){
				fix16_boxes[total_box_count].confidence = max_score;
				fix16_boxes[total_box_count].class_id = max_score_ind;
				fix16_boxes[total_box_count].xmin = fix16_to_int((output[i*output_sz+0] - fix16_mul(output[i*output_sz+2],fix16_half))*input_w);
				fix16_boxes[total_box_count].xmax = fix16_to_int((output[i*output_sz+0] + fix16_mul(output[i*output_sz+2],fix16_half))*input_w);
				fix16_boxes[total_box_count].ymin = fix16_to_int((output[i*output_sz+1] - fix16_mul(output[i*output_sz+3],fix16_half))*input_h);
				fix16_boxes[total_box_count].ymax = fix16_to_int((output[i*output_sz+1] + fix16_mul(output[i*output_sz+3],fix16_half))*input_h);

				fix16_boxes[total_box_count].x = fix16_to_int((output[i*output_sz+0])*input_w);
				fix16_boxes[total_box_count].w = fix16_to_int((output[i*output_sz+2])*input_w);
				fix16_boxes[total_box_count].y = fix16_to_int((output[i*output_sz+1])*input_h);
				fix16_boxes[total_box_count].h = fix16_to_int((output[i*output_sz+3])*input_h);
				if (is_obb) {
					fix16_boxes[total_box_count].angle = output[i*output_sz+4+num_classes];
				} else {
					fix16_boxes[total_box_count].angle = 0;
				}

				if (is_pose) {
					for(int p = 0; p <17;p++){
						poses[total_box_count].keypoints[p][0] = fix16_mul(output[i*output_sz+4+num_classes+p*3+ 0], fix16_from_int(input_w));
						poses[total_box_count].keypoints[p][1] = fix16_mul(output[i*output_sz+4+num_classes+p*3+ 1], fix16_from_int(input_h));
						poses[total_box_count].scores[p] = output[i*output_sz+4+num_classes+p*3+ 2];
					}

				}
				total_box_count++;

			}
			if(total_box_count>=boxes_len)
				break;
		}
		fix16_sort_boxes(fix16_boxes, poses, total_box_count);
		if (is_pose) {
			fix16_do_nmm(fix16_boxes, poses, total_box_count, overlap);
		} else {
			fix16_do_nms(fix16_boxes, total_box_count, overlap);
		}
		int clean_box_count = fix16_clean_boxes(fix16_boxes, poses, total_box_count, input_w, input_h);
		return clean_box_count;
}

int post_process_yolo(fix16_t **outputs, const int num_outputs, yolo_info_t *cfg,
		fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes)
{
	int total_box_count = 0;
	int input_h  = cfg[0].input_dims[1];
	int input_w  = cfg[0].input_dims[2];
	fix16_t *anchors = cfg[0].anchors;

	fix16_t fix16_log_odds = fix16_log(fix16_div(thresh, fix16_sub(fix16_one, thresh)));
	for (int o = 0; o < num_outputs; o++) {
		fix16_t *out32 = outputs[o];
		int num_per_output = cfg[o].num;
		if (num_outputs > 1) num_per_output = cfg[o].mask_length;

		
		fix16_t fix16_biases[2*num_per_output];

		int h  = cfg[o].output_dims[1];
		int w  = cfg[o].output_dims[2];
		fix16_t h_ratio = fix16_from_int(input_h/h);
		fix16_t w_ratio = fix16_from_int(input_w/w);

		for (int i = 0; i < num_per_output; i++) {
			int mask = i;
			if (num_outputs > 1) mask = cfg[o].mask[i];

			if (cfg[o].version == 2) {
				fix16_biases[2*i] = fix16_mul(anchors[2*mask], w_ratio);
				fix16_biases[2*i+1] = fix16_mul(anchors[2*mask+1], h_ratio);
			} else {
				fix16_biases[2*i] = anchors[2*mask];
				fix16_biases[2*i+1] = anchors[2*mask+1];
			}
		}
		int fix16_box_count = fix16_get_region_boxes(out32, fix16_biases, w, h, num_per_output, cfg[o].classes, w_ratio,
				h_ratio, thresh, fix16_log_odds,
				fix16_boxes + total_box_count,max_boxes-total_box_count, 1, 1, cfg[o].version);
		fflush(stdout);

		// copy boxes
		total_box_count += fix16_box_count;
		if(total_box_count == max_boxes){
			break;
		}
	}

	fix16_sort_boxes(fix16_boxes, NULL, total_box_count);
	fix16_do_nms(fix16_boxes, total_box_count, overlap);

	int clean_box_count = fix16_clean_boxes(fix16_boxes, NULL, total_box_count, input_w, input_h);

	return clean_box_count;
}

void post_process_classifier(fix16_t *outputs, const int out_sz, int16_t* output_index, int topk)
{

	fix16_t* cached_output=malloc(out_sz*sizeof(*cached_output));
	for(int i=0;i<out_sz;++i){
		cached_output[i] = outputs[i];
	}
	post_classifier(cached_output, out_sz, output_index, topk);
	free(cached_output);
}

void post_process_classifier_int8(int8_t *outputs, const int out_sz, int16_t* output_index, int topk)
{
	int8_t* cached_output=malloc(out_sz*sizeof(*cached_output));
	for(int i=0;i<out_sz;++i){
		cached_output[i] = outputs[i];
	}
	post_classifier_int8(cached_output, out_sz, output_index, topk);
	free(cached_output);
}

void ctc_raw_indices(int *indices, fix16_t *output, const int output_len, const int output_depth)
{
    for (int x = 0; x < output_len; x++){
	    int max_idx = 0;
	    fix16_t max = -1;

	    for (int y = 0; y < output_depth; y++){
		    if (output[y*output_len + x] > max) {
			    max = output[y*output_len + x];
			    max_idx = y;
		    }
	    }

	    if (max > 0) {
		    indices[x] = max_idx;
	    } else {
		    indices[x] = -1;
	    }
    }
}


void ctc_greedy_decode(int *unique, fix16_t *output, const int output_len, const int output_depth, int merge_repeated)
{
    int blank_index = output_depth - 1;
    int indices[output_len]; 
    ctc_raw_indices(indices, output, output_len, output_depth);

    int idx, prev = -1, uidx = 0;
    for (int i = 0; i < output_len; i++){
	    unique[i] = -1;
	    idx = indices[i];
	    if (idx != blank_index && idx != -1) {
		    if(merge_repeated) { 
			if (prev == -1 || prev != idx) {
			    prev = idx;
			    unique[uidx]=idx;
			    uidx++;
			}
		    } else {
			    unique[uidx]=idx;
			    uidx++;
		    }
		    
	    } else {
		    prev = -1;
	    }
    }
}


#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif


static const unsigned char blazeAnchors256[896][2] =
{{8,8},{8,8},{24,8},{24,8},{40,8},{40,8},{56,8},{56,8},
	{72,8},{72,8},{88,8},{88,8},{104,8},{104,8},{120,8},{120,8},{136,8},{136,8},{152,8},{152,8},{168,8},{168,8},{184,8},{184,8},
	{200,8},{200,8},{216,8},{216,8},{232,8},{232,8},{248,8},{248,8},{8,24},{8,24},{24,24},{24,24},{40,24},{40,24},{56,24},{56,24},
	{72,24},{72,24},{88,24},{88,24},{104,24},{104,24},{120,24},{120,24},{136,24},{136,24},{152,24},{152,24},{168,24},{168,24},{184,24},{184,24},
	{200,24},{200,24},{216,24},{216,24},{232,24},{232,24},{248,24},{248,24},{8,40},{8,40},{24,40},{24,40},{40,40},{40,40},{56,40},{56,40},
	{72,40},{72,40},{88,40},{88,40},{104,40},{104,40},{120,40},{120,40},{136,40},{136,40},{152,40},{152,40},{168,40},{168,40},{184,40},{184,40},
	{200,40},{200,40},{216,40},{216,40},{232,40},{232,40},{248,40},{248,40},{8,56},{8,56},{24,56},{24,56},{40,56},{40,56},{56,56},{56,56},
	{72,56},{72,56},{88,56},{88,56},{104,56},{104,56},{120,56},{120,56},{136,56},{136,56},{152,56},{152,56},{168,56},{168,56},{184,56},{184,56},
	{200,56},{200,56},{216,56},{216,56},{232,56},{232,56},{248,56},{248,56},{8,72},{8,72},{24,72},{24,72},{40,72},{40,72},{56,72},{56,72},
	{72,72},{72,72},{88,72},{88,72},{104,72},{104,72},{120,72},{120,72},{136,72},{136,72},{152,72},{152,72},{168,72},{168,72},{184,72},{184,72},
	{200,72},{200,72},{216,72},{216,72},{232,72},{232,72},{248,72},{248,72},{8,88},{8,88},{24,88},{24,88},{40,88},{40,88},{56,88},{56,88},
	{72,88},{72,88},{88,88},{88,88},{104,88},{104,88},{120,88},{120,88},{136,88},{136,88},{152,88},{152,88},{168,88},{168,88},{184,88},{184,88},
	{200,88},{200,88},{216,88},{216,88},{232,88},{232,88},{248,88},{248,88},{8,104},{8,104},{24,104},{24,104},{40,104},{40,104},{56,104},{56,104},
	{72,104},{72,104},{88,104},{88,104},{104,104},{104,104},{120,104},{120,104},{136,104},{136,104},{152,104},{152,104},{168,104},{168,104},{184,104},{184,104},
	{200,104},{200,104},{216,104},{216,104},{232,104},{232,104},{248,104},{248,104},{8,120},{8,120},{24,120},{24,120},{40,120},{40,120},{56,120},{56,120},
	{72,120},{72,120},{88,120},{88,120},{104,120},{104,120},{120,120},{120,120},{136,120},{136,120},{152,120},{152,120},{168,120},{168,120},{184,120},{184,120},
	{200,120},{200,120},{216,120},{216,120},{232,120},{232,120},{248,120},{248,120},{8,136},{8,136},{24,136},{24,136},{40,136},{40,136},{56,136},{56,136},
	{72,136},{72,136},{88,136},{88,136},{104,136},{104,136},{120,136},{120,136},{136,136},{136,136},{152,136},{152,136},{168,136},{168,136},{184,136},{184,136},
	{200,136},{200,136},{216,136},{216,136},{232,136},{232,136},{248,136},{248,136},{8,152},{8,152},{24,152},{24,152},{40,152},{40,152},{56,152},{56,152},
	{72,152},{72,152},{88,152},{88,152},{104,152},{104,152},{120,152},{120,152},{136,152},{136,152},{152,152},{152,152},{168,152},{168,152},{184,152},{184,152},
	{200,152},{200,152},{216,152},{216,152},{232,152},{232,152},{248,152},{248,152},{8,168},{8,168},{24,168},{24,168},{40,168},{40,168},{56,168},{56,168},
	{72,168},{72,168},{88,168},{88,168},{104,168},{104,168},{120,168},{120,168},{136,168},{136,168},{152,168},{152,168},{168,168},{168,168},{184,168},{184,168},
	{200,168},{200,168},{216,168},{216,168},{232,168},{232,168},{248,168},{248,168},{8,184},{8,184},{24,184},{24,184},{40,184},{40,184},{56,184},{56,184},
	{72,184},{72,184},{88,184},{88,184},{104,184},{104,184},{120,184},{120,184},{136,184},{136,184},{152,184},{152,184},{168,184},{168,184},{184,184},{184,184},
	{200,184},{200,184},{216,184},{216,184},{232,184},{232,184},{248,184},{248,184},{8,200},{8,200},{24,200},{24,200},{40,200},{40,200},{56,200},{56,200},
	{72,200},{72,200},{88,200},{88,200},{104,200},{104,200},{120,200},{120,200},{136,200},{136,200},{152,200},{152,200},{168,200},{168,200},{184,200},{184,200},
	{200,200},{200,200},{216,200},{216,200},{232,200},{232,200},{248,200},{248,200},{8,216},{8,216},{24,216},{24,216},{40,216},{40,216},{56,216},{56,216},
	{72,216},{72,216},{88,216},{88,216},{104,216},{104,216},{120,216},{120,216},{136,216},{136,216},{152,216},{152,216},{168,216},{168,216},{184,216},{184,216},
	{200,216},{200,216},{216,216},{216,216},{232,216},{232,216},{248,216},{248,216},{8,232},{8,232},{24,232},{24,232},{40,232},{40,232},{56,232},{56,232},
	{72,232},{72,232},{88,232},{88,232},{104,232},{104,232},{120,232},{120,232},{136,232},{136,232},{152,232},{152,232},{168,232},{168,232},{184,232},{184,232},
	{200,232},{200,232},{216,232},{216,232},{232,232},{232,232},{248,232},{248,232},{8,248},{8,248},{24,248},{24,248},{40,248},{40,248},{56,248},{56,248},
	{72,248},{72,248},{88,248},{88,248},{104,248},{104,248},{120,248},{120,248},{136,248},{136,248},{152,248},{152,248},{168,248},{168,248},{184,248},{184,248},
	{200,248},{200,248},{216,248},{216,248},{232,248},{232,248},{248,248},{248,248},{16,16},{16,16},{16,16},{16,16},{16,16},{16,16},{48,16},{48,16},
	{48,16},{48,16},{48,16},{48,16},{80,16},{80,16},{80,16},{80,16},{80,16},{80,16},{112,16},{112,16},{112,16},{112,16},{112,16},{112,16},
	{144,16},{144,16},{144,16},{144,16},{144,16},{144,16},{176,16},{176,16},{176,16},{176,16},{176,16},{176,16},{208,16},{208,16},{208,16},{208,16},
	{208,16},{208,16},{240,16},{240,16},{240,16},{240,16},{240,16},{240,16},{16,48},{16,48},{16,48},{16,48},{16,48},{16,48},{48,48},{48,48},
	{48,48},{48,48},{48,48},{48,48},{80,48},{80,48},{80,48},{80,48},{80,48},{80,48},{112,48},{112,48},{112,48},{112,48},{112,48},{112,48},
	{144,48},{144,48},{144,48},{144,48},{144,48},{144,48},{176,48},{176,48},{176,48},{176,48},{176,48},{176,48},{208,48},{208,48},{208,48},{208,48},
	{208,48},{208,48},{240,48},{240,48},{240,48},{240,48},{240,48},{240,48},{16,80},{16,80},{16,80},{16,80},{16,80},{16,80},{48,80},{48,80},
	{48,80},{48,80},{48,80},{48,80},{80,80},{80,80},{80,80},{80,80},{80,80},{80,80},{112,80},{112,80},{112,80},{112,80},{112,80},{112,80},
	{144,80},{144,80},{144,80},{144,80},{144,80},{144,80},{176,80},{176,80},{176,80},{176,80},{176,80},{176,80},{208,80},{208,80},{208,80},{208,80},
	{208,80},{208,80},{240,80},{240,80},{240,80},{240,80},{240,80},{240,80},{16,112},{16,112},{16,112},{16,112},{16,112},{16,112},{48,112},{48,112},
	{48,112},{48,112},{48,112},{48,112},{80,112},{80,112},{80,112},{80,112},{80,112},{80,112},{112,112},{112,112},{112,112},{112,112},{112,112},{112,112},
	{144,112},{144,112},{144,112},{144,112},{144,112},{144,112},{176,112},{176,112},{176,112},{176,112},{176,112},{176,112},{208,112},{208,112},{208,112},{208,112},
	{208,112},{208,112},{240,112},{240,112},{240,112},{240,112},{240,112},{240,112},{16,144},{16,144},{16,144},{16,144},{16,144},{16,144},{48,144},{48,144},
	{48,144},{48,144},{48,144},{48,144},{80,144},{80,144},{80,144},{80,144},{80,144},{80,144},{112,144},{112,144},{112,144},{112,144},{112,144},{112,144},
	{144,144},{144,144},{144,144},{144,144},{144,144},{144,144},{176,144},{176,144},{176,144},{176,144},{176,144},{176,144},{208,144},{208,144},{208,144},{208,144},
	{208,144},{208,144},{240,144},{240,144},{240,144},{240,144},{240,144},{240,144},{16,176},{16,176},{16,176},{16,176},{16,176},{16,176},{48,176},{48,176},
	{48,176},{48,176},{48,176},{48,176},{80,176},{80,176},{80,176},{80,176},{80,176},{80,176},{112,176},{112,176},{112,176},{112,176},{112,176},{112,176},
	{144,176},{144,176},{144,176},{144,176},{144,176},{144,176},{176,176},{176,176},{176,176},{176,176},{176,176},{176,176},{208,176},{208,176},{208,176},{208,176},
	{208,176},{208,176},{240,176},{240,176},{240,176},{240,176},{240,176},{240,176},{16,208},{16,208},{16,208},{16,208},{16,208},{16,208},{48,208},{48,208},
	{48,208},{48,208},{48,208},{48,208},{80,208},{80,208},{80,208},{80,208},{80,208},{80,208},{112,208},{112,208},{112,208},{112,208},{112,208},{112,208},
	{144,208},{144,208},{144,208},{144,208},{144,208},{144,208},{176,208},{176,208},{176,208},{176,208},{176,208},{176,208},{208,208},{208,208},{208,208},{208,208},
	{208,208},{208,208},{240,208},{240,208},{240,208},{240,208},{240,208},{240,208},{16,240},{16,240},{16,240},{16,240},{16,240},{16,240},{48,240},{48,240},
	{48,240},{48,240},{48,240},{48,240},{80,240},{80,240},{80,240},{80,240},{80,240},{80,240},{112,240},{112,240},{112,240},{112,240},{112,240},{112,240},
	{144,240},{144,240},{144,240},{144,240},{144,240},{144,240},{176,240},{176,240},{176,240},{176,240},{176,240},{176,240},{208,240},{208,240},{208,240},{208,240},
	{208,240},{208,240},{240,240},{240,240},{240,240},{240,240},{240,240},{240,240}};


static fix16_t calcIou(fix16_t* A, fix16_t* B){
	// pointers to elements (x, y, width, height)
	fix16_t left = MAX(A[0] - (A[2]>>1), B[0] - (B[2]>>1));
	fix16_t right = MIN(A[0] + (A[2]>>1), B[0] + (B[2]>>1));
	fix16_t top = MAX(A[1] - (A[3]>>1), B[1] - (B[3]>>1));
	fix16_t bottom = MIN(A[1] + (A[3]>>1), B[1] + (B[3]>>1));
	fix16_t i = fix16_mul(MAX(0,right-left), MAX(0,bottom-top));    // intersection
	fix16_t u = fix16_mul(A[2], A[3]) + fix16_mul(B[2], B[3]) - i;  // union
	if(u>0)
		return fix16_div(i, u);
	else
		return 0;
}

int post_process_blazeface(object_t faces[],fix16_t* scores,fix16_t* points,int scoresLength,int max_faces, fix16_t anchorsScale) {

	fix16_t min_suppression_threshold = F16(0.3);
	fix16_t raw_thresh = F16(1.0986122886681098);    // 1.0986122886681098 == -log((1-thresh)/thresh),  thresh=0.75

	int num_detects = 0;   // number of detects (before combining by blending)
	const int maxRawDetects = 64;
	int ind[maxRawDetects];    // indices of scores in descending order

	// find detections based on score and add to a sorted list of indices (indices of highest scores first)
	for(int n=0; n<scoresLength; n++){
		if(scores[n] > raw_thresh){
			scores[n] = fix16_div(fix16_one, fix16_one + fix16_exp(-scores[n]));    // compute score
			int i=0;
			while(i<num_detects){ // find the insertion index
				if(scores[n] > scores[ind[i]]){
					for(int i2=MIN(num_detects,maxRawDetects-1); i2>i; i2--) // move down all lower elements
						ind[i2] = ind[i2-1];
					ind[i] = n;
					num_detects++;
					break;
				}
				i++;
			}
			if(i==num_detects && num_detects<maxRawDetects){   // if not inserted and there's room, then insert at the end
				ind[i] = n;
				num_detects++;
			}
		}
	}


	// add achors to points (omitting indices 2 and 3, which are the width and height of the box)
	for(int i=0; i<num_detects; i++){
		int n = ind[i];
		fix16_t xAnchor = fix16_mul(fix16_from_int(blazeAnchors256[n][0]),anchorsScale);
		fix16_t yAnchor = fix16_mul(fix16_from_int(blazeAnchors256[n][1]),anchorsScale);
		points[n*16 + 0] += xAnchor;
		points[n*16 + 1] += yAnchor;
		for(int p=4; p<16; p+=2){
			points[n*16 + p] += xAnchor;
			points[n*16 + p+1] += yAnchor;
		}
	}

	// find overlapping detects and blend them together, then add to array of face structures
	int facesLength = 0;
	char used[num_detects];   // true if raw detect was blended with another detect (make sure it doesn't get used again)
	for(int i=0; i<num_detects; i++)
		used[i] = 0;
	for(int i1=0; i1<num_detects; i1++){
		if(used[i1])
			continue;
		fix16_t totalScore = scores[ind[i1]];
		fix16_t blendPoints[16];
		for(int p=0; p<16; p++)
			blendPoints[p] = fix16_mul(points[ind[i1]*16+p], scores[ind[i1]]);  // weight based on score
		for(int i2=i1+1; i2<num_detects; i2++){
			if(used[i2])
				continue;
			fix16_t iou = calcIou(&points[ind[i1]*16], &points[ind[i2]*16]);
			if(iou > min_suppression_threshold){
				used[i2] = 1;
				for(int p=0; p<16; p++)
					blendPoints[p] += fix16_mul(scores[ind[i2]], points[ind[i2]*16+p]);
				totalScore += scores[ind[i2]];
			}
		}
		fix16_t scale = fix16_div(fix16_one, totalScore);
		for(int p=0; p<16; p++)
			blendPoints[p] = fix16_mul(blendPoints[p], scale);   // scale back to original

		fix16_t x = blendPoints[0];
		fix16_t y = blendPoints[1];
		fix16_t w = blendPoints[2];
		fix16_t h = blendPoints[3];
		blendPoints[0] = x-w/2;   // convert from (x,y,w,h) to (left,top,right,bottom)
		blendPoints[1] = y-h/2;
		blendPoints[2] = x+w/2;
		blendPoints[3] = y+h/2;
		// write to class face struct
		for(int p=0; p<4; p++)
			faces[facesLength].box[p] = blendPoints[p];
		for(int p=0; p<4; p++){
			faces[facesLength].points[p][0] = blendPoints[p*2+4];
			faces[facesLength].points[p][1] = blendPoints[p*2+5];
		}
		faces[facesLength].detect_score = scores[ind[i1]];
		facesLength++;
		if(facesLength==max_faces)
			break;
	}
	return facesLength;
}

int pprint_post_process(const char *name, const char *pptype, model_t *model, fix16_t **o_buffers,int int8_flag, int fps)
{
	char label[256];
	const int topk=5;
	int *in_dims = model_get_input_shape(model,0);
	int total_dims = model_get_input_dims(model,0);
	int num_outputs = (int)model_get_num_outputs(model);
	int input_h = in_dims[total_dims-2];
	int input_w = in_dims[total_dims-1];
	
#ifdef HARDWARE_DRAW
	fix16_t hratio = fix16_div(fix16_from_int(1080),fix16_from_int(input_h));
	fix16_t wratio = fix16_div(fix16_from_int(1920),fix16_from_int(input_w));
	
	snprintf(label,sizeof(label),"%s %dx%d  %d fps",name,input_w,input_h, fps);
	draw_label(label,0,2,overlay_draw_frame,2048,1080,WHITE);
#else
	topk_draw = topk;
#endif	
	int coords = 4;
	int classes = 80;
	int mask_length = 3;
	int num = 3 * num_outputs;
	int anchors_length = 2 * num;
	if (!strcmp(pptype, "BLAZEFACE")){
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		// reverse
		fix16_t* output_buffer0=(fix16_t*)(uintptr_t)o_buffers[1];
		fix16_t* output_buffer1=(fix16_t*)(uintptr_t)o_buffers[0];
		int output_length0 = model_get_output_length(model, 1);
		int output_length1 = model_get_output_length(model, 0);

		int facesLength = 0;
		if (output_length0 < output_length1) {
			facesLength = post_process_blazeface(faces,output_buffer0,output_buffer1,output_length0,
					MAX_FACES,fix16_from_int(1));
		} else {
			facesLength = post_process_blazeface(faces,output_buffer1,output_buffer0,output_length1,
					MAX_FACES,fix16_from_int(1));
		}
		for(int f=0;f<facesLength;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
			printf("face %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
		}
	} else if (!strcmp(pptype,"RETINAFACE")){
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		fix16_t confidence_threshold=F16(0.8);
		fix16_t nms_threshold=F16(0.4);

		fix16_t* output_buffers[9];
		//( 0 1 2 3 4 5 6 7 8)->(5 4 3 8 7 6 2 1 0)
		output_buffers[0]=(fix16_t*)(uintptr_t)o_buffers[5];
		output_buffers[1]=(fix16_t*)(uintptr_t)o_buffers[4];
		output_buffers[2]=(fix16_t*)(uintptr_t)o_buffers[3];
		output_buffers[3]=(fix16_t*)(uintptr_t)o_buffers[8];
		output_buffers[4]=(fix16_t*)(uintptr_t)o_buffers[7];
		output_buffers[5]=(fix16_t*)(uintptr_t)o_buffers[6];
		output_buffers[6]=(fix16_t*)(uintptr_t)o_buffers[2];
		output_buffers[7]=(fix16_t*)(uintptr_t)o_buffers[1];
		output_buffers[8]=(fix16_t*)(uintptr_t)o_buffers[0];


		int facesLength = post_process_retinaface(faces,MAX_FACES,output_buffers, input_w, input_h,
				confidence_threshold,nms_threshold);

		for(int f=0;f<facesLength;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
			printf("face %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
			printf("landmarks: ");
			for(int l =0;l<5;++l){
				printf("%3.1f,%3.1f ",
						fix16_to_float(face->points[l][0]),
						fix16_to_float(face->points[l][1]));
				fflush(stdout);
			}printf("\n");


		}
	} else if (!strcmp(pptype,"LPD")) {
		int platesLength = 0;
		const int MAX_PLATES=10;
		object_t plates[MAX_PLATES];
		fix16_t confidence_threshold=F16(0.55);
		fix16_t nms_threshold=F16(0.2);		

		fix16_t* fix16_buffers[9];
		int8_t* output_buffer_int8[9];
		int zero_points[9];
		fix16_t scale_outs[9];


		for (int o = 0; o < num_outputs; o++) {				
			int *output_shape = model_get_output_shape(model,o);
			int ind = 2*(output_shape[2]/18) + (output_shape[1]/6); 
			fix16_buffers[ind]=(fix16_t*)(uintptr_t)o_buffers[o]; //assigns output buffers by first dim ascending, second descending
			output_buffer_int8[ind]= (int8_t*)(uintptr_t)o_buffers[o];
			zero_points[ind]=model_get_output_zeropoint(model,o);
			scale_outs[ind]=model_get_output_scale_fix16_value(model,o);
		}
		if(int8_flag){
			platesLength = post_process_lpd_int8(plates, MAX_PLATES, output_buffer_int8, input_w, input_h,
				confidence_threshold,nms_threshold, num_outputs,zero_points,scale_outs);
		}
		else{
			platesLength = post_process_lpd(plates, MAX_PLATES, fix16_buffers, input_w, input_h,
				confidence_threshold,nms_threshold, num_outputs);
		}
		for(int f=0;f<platesLength;f++){
			object_t* plate = plates+f;
			fix16_t x = plate->box[0];
			fix16_t y = plate->box[1];
			fix16_t w = plate->box[2];
			fix16_t h = plate->box[3];
#ifdef HARDWARE_DRAW
			x = fix16_sub(x, fix16_div(w, fix16_two));
			y = fix16_sub(y, fix16_div(h, fix16_two));
			if( x > 0 &&  y > 0 && w > 0 && h > 0) {
				x = fix16_mul(x, wratio);
				y = fix16_mul(y, hratio);
				w = fix16_mul(w, wratio);
				h = fix16_mul(h, hratio);
				draw_box(fix16_to_int(x),
						fix16_to_int(y)+540,
						fix16_to_int(w),
						fix16_to_int(h),
						5,
						get_colour_modulo(0),
						overlay_draw_frame,2048,1080);
			}
#else			
			printf("plate %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
#endif

		}
	} else if (!strcmp(pptype, "LPR")){
		fix16_t conf = 0;
		fix16_t* fix16_buffers = (fix16_t*)(uintptr_t)o_buffers[0];
		int8_t* output_buffer_int8 = (int8_t*)(uintptr_t)o_buffers[0];
		if(int8_flag){
			conf = post_process_lpr_int8(output_buffer_int8, model, label);
		}
		else{
			conf = post_process_lpr(fix16_buffers, model_get_output_length(model, 0), label);
		}
		printf("Plate ID: %s Recognition Score: %3.4f\n", label, fix16_to_float(conf));

	} else if (!strcmp(pptype,"SCRFD")) {
		int facesLength = 0;
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		fix16_t confidence_threshold=F16(0.8);
		fix16_t nms_threshold=F16(0.4);
		fix16_t* fix16_buffers[9];
		int8_t* output_buffer_int8[9];
		int zero_points[9];
		fix16_t scale_outs[9];
		
		for(int o=0; o<num_outputs; o++){
			int *output_shape = model_get_output_shape(model,o);
			int ind = (output_shape[1]/8)*3 + (2-(output_shape[2]/18)); //first dim should be {2,8,20} second dim should be {9,18,36}
			fix16_buffers[ind]=(fix16_t*)(uintptr_t)o_buffers[o]; //assigns output buffers by first dim ascending, second descending
			output_buffer_int8[ind]= (int8_t*)(uintptr_t)o_buffers[o];
			zero_points[ind]=model_get_output_zeropoint(model,o);
			scale_outs[ind]=model_get_output_scale_fix16_value(model,o);
		}
		if(int8_flag){
			facesLength = post_process_scrfd_int8(faces,MAX_FACES,output_buffer_int8, zero_points, scale_outs, input_w, input_h,
				confidence_threshold,nms_threshold,model);
		}	

		else{			
			facesLength = post_process_scrfd(faces, MAX_FACES, fix16_buffers, input_w, input_h,
				confidence_threshold,nms_threshold);
		}
		for(int f=0;f<facesLength;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
#ifdef HARDWARE_DRAW
			if( x > 0 &&  y > 0 && w > 0 && h > 0) {
				x = fix16_mul(x, wratio);
				y = fix16_mul(y, hratio);
				w = fix16_mul(w, wratio);
				h = fix16_mul(h, hratio);
				draw_box(fix16_to_int(x),
						fix16_to_int(y),
						fix16_to_int(w),
						fix16_to_int(h),
						5,
						get_colour_modulo(0),
						overlay_draw_frame,2048,1080);
			}
#else
			printf("face %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f w/ conf: %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h),fix16_to_float(face->detect_score));
			printf("landmarks: ");
			for(int l =0;l<5;++l){
				printf("%3.1f,%3.1f ",
						fix16_to_float(face->points[l][0]),
						fix16_to_float(face->points[l][1]));
				fflush(stdout);
			}printf("\n");	
#endif

		}
	} else if (!strcmp(pptype, "CLASSIFY")){

		int output_length = model_get_output_length(model, 0);
		fix16_t* output_buffer0=(fix16_t*)(uintptr_t)o_buffers[0];
		int8_t* output_buffer_int8_0=(int8_t*)(uintptr_t)o_buffers[0];
		fix16_t f16_scale = (fix16_t)model_get_output_scale_fix16_value(model,0); // get output scale
		int32_t zero_point = model_get_output_zeropoint(model,0); // get output zero
		if(int8_flag){
			post_process_classifier_int8(output_buffer_int8_0,output_length,indexes,topk);
		}
		else{
			post_process_classifier(output_buffer0,output_length,indexes,topk);
		}
		char* class_name = "";
		if(update_Classifier)
			for(int i=0;i<topk_draw;++i) {
				int idx = indexes[i];
				display_index[i]=indexes[i];
				scores[i] = fix16_mul(fix16_from_int((int32_t)(output_buffer_int8_0[idx])-zero_point),f16_scale);				
			}		
		for(int i = 0;i < topk_draw; ++i){
			int idx = display_index[i];
			int32_t score = scores[i];
			if(output_length == 1001 || output_length == 1000){ // imagenet
				class_name = imagenet_classes[idx];
				if(output_length==1001){
					//some imagenet networks have a null catagory, account for that
					class_name =  imagenet_classes[idx-1];
				}
			} 
#ifdef HARDWARE_DRAW
			snprintf(label,sizeof(label),"%d %s %d%%",i,class_name,(score*100)>>16);
			draw_label(label,20,36+i*34,overlay_draw_frame,2048,1080,WHITE);
#else
			printf("%d: %d - %s = %3.2f\n", i, idx, class_name, fix16_to_float(score));
#endif
		}
	} else if (!strcmp(pptype, "PLATE")){
		fix16_t* output=(fix16_t*)(uintptr_t)o_buffers[0];
		char **lpr = NULL;
		int output_len, output_depth;
		int merge_repeated = 1;
		char *is_chinese = strstr(name, "arrier");
		if (is_chinese == NULL) is_chinese = strstr(name, "ARRIER");

		if (is_chinese) {
			lpr = lpr_chinese_characters;
			output_depth = 71;
			output_len = 88;
		} else {
			lpr = lpr_characters;
			output_depth = 37;
			output_len = 106;
		}
		int unique[output_len];
		ctc_greedy_decode(unique, output, output_len, output_depth, merge_repeated);

		int i = 0;
		while(unique[i] != -1) {
			printf("'%s'", lpr[unique[i]]);
			i++;
		}
		printf("\n");


	} else if (!strcmp(pptype, "YOLOV2") || !strcmp(pptype, "YOLOV3") || !strcmp(pptype, "YOLOV4") || !strcmp(pptype, "YOLOV5") || !strcmp(pptype, "SSDV2") || !strcmp(pptype, "SSDTORCH") || !strcmp(pptype, "ULTRALYTICS_FULL") || !strcmp(pptype, "ULTRALYTICS")) {
		char **class_names = NULL;
		int valid_boxes = 0;
		int max_boxes = 100;
		const int boxes_len = 1024;
		fix16_box boxes[boxes_len];
		fix16_t thresh = F16(0.3);
		fix16_t iou = F16(0.4);

		int is_tiny = strstr(name, "iny")? 1 : 0;
		if (is_tiny == 0) is_tiny = strstr(name, "INY")? 1 : 0;

		if(!strcmp(pptype, "YOLOV2")){ //tiny yolo v2
			fix16_t *outputs[] = {(fix16_t*)(uintptr_t)o_buffers[0]};
			int8_t *output_int8[] = {(int8_t*)(uintptr_t)o_buffers[0]};
			int zero_point[] = {model_get_output_zeropoint(model,0)};
			fix16_t f16_scale[] = {model_get_output_scale_fix16_value(model,0)};
			int* oshape = model_get_output_shape(model,0);
			
			fix16_t tiny_anchors[] ={F16(1.08),F16(1.19),F16(3.42),F16(4.41),F16(6.63),F16(11.38),F16(9.42),F16(5.11),F16(16.620001),F16(10.52)};
			fix16_t anchors[] = {F16(1.3221),F16(1.73145),F16(3.19275),F16(4.00944),F16(5.05587),F16(8.09892),F16(9.47112),F16(4.84053),F16(11.2364),F16(10.0071)};		
			thresh = F16(0.6);
			num = 5;
			anchors_length = 2*num;

			if (oshape[1] == 125) { // yolo v2 voc (4+1+classes)*num
				class_names = voc_classes;
				classes = 20;
			} else if (oshape[1] == 425){ // yolo v2 coco
				class_names = coco_classes;
				memcpy(anchors,(fix16_t[]){F16(0.57273),F16(0.677385),F16(1.87446),F16(2.06253),F16(3.33843),F16(5.47434),F16(7.88282),F16(3.52778),F16(9.77052),F16(9.16828)},anchors_length*sizeof(fix16_t));
			} 
			yolo_info_t cfg_0 = {
				.version = 2,
				.input_dims = {in_dims[1], in_dims[2], in_dims[3]},
				.output_dims = {oshape[1], oshape[2], oshape[3]},
				.coords = coords,
				.classes = classes,
				.num = num,
				.anchors_length = anchors_length,
				.anchors = (is_tiny==1) ? tiny_anchors : anchors,
			};
			
			yolo_info_t cfg[] = {cfg_0};
			if(int8_flag){
				valid_boxes = post_process_yolo_int8(output_int8, num_outputs, zero_point, f16_scale, cfg, thresh, iou, boxes, max_boxes);
			}
			else{
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}

		} else if (!strcmp(pptype, "ULTRALYTICS_FULL")){
			class_names = coco_classes;
			fix16_t* output=(fix16_t*)(uintptr_t)o_buffers[0];
			int8_t* output_int8 =(int8_t*)(uintptr_t)o_buffers[0];
			fix16_t f16_scale = (fix16_t)model_get_output_scale_fix16_value(model,0); // get output scale
			int32_t zero_point = model_get_output_zeropoint(model,0); // get output zero
			if(int8_flag){
				valid_boxes = post_process_ultra_nms_int8(output_int8, 8400, input_h, input_w,f16_scale,zero_point, thresh, iou, boxes, max_boxes, 80);
			} else{
				valid_boxes = post_process_ultra_nms(output, 8400, input_h, input_w, thresh, iou, boxes, NULL, max_boxes, 80, 0, 0);
			}

		} else if (!strcmp(pptype, "ULTRALYTICS")){
			class_names = coco_classes;
			int* outputs_shape[9];
			int8_t *outputs_int8[9];
			int zero_points[9];
			fix16_t scale_outs[9];

			// put outputs in this order
			// type:  {class_stride8, box_stride8,   class_stride16,  box_stride16,    class_stride32,  box_stride32}
			// shape: {[1,80,H/8,W/8],[1,64,H/8,W/8],[1,80,H/16,W/16],[1,64,H/16,W/16],[1,80,H/32,W/32],[1,64,H/32,W/32]}
			int32_t w_min = 0x7FFFFFFF;	// minimum width must be stride32
			int32_t w_max = 0;			// maximum width must be stride8
			int* shapes[9];
			
			for(int n=0; n<num_outputs; n++){
				shapes[n] = model_get_output_shape(model,n);
				w_min = MIN(shapes[n][3], w_min);
				w_max = MAX(shapes[n][3], w_max);
			}

			for(int i=0; i<num_outputs; i++){
				int o=2;	// proper order
				if(shapes[i][3]==w_min) o=4;		// stride 8
				else if(shapes[i][3]==w_max) o=0;	// stride 32
				else o=2;							// stride 16
				if(shapes[i][1]==64) o+=1;
				else if(shapes[i][1] == 1) o= o/2 + 6;			// box (otherwise class)
				outputs_shape[o] = shapes[i];
				outputs_int8[o] = (int8_t*)(uintptr_t)o_buffers[i];
				zero_points[o]=model_get_output_zeropoint(model,i);
				scale_outs[o]=model_get_output_scale_fix16_value(model,i);
			}			
			
			const int max_detections = 200;
			fix16_t post_buffer[max_detections*(4+80)];
			int post_len;
			post_len = post_process_ultra_int8(outputs_int8, outputs_shape, post_buffer, thresh, zero_points, scale_outs, max_detections, 0, 0,num_outputs);
			valid_boxes = post_process_ultra_nms(post_buffer, post_len, input_h, input_w, thresh, iou, boxes, NULL, boxes_len, 80, 0, 0);

		} else if (!strcmp(pptype, "YOLOV3") || !strcmp(pptype, "YOLOV4")){ //tiny yolo v3/v4 COCO
			class_names = coco_classes;
			fix16_t *outputs[3];
			int8_t *outputs_int8[3];
			int zero_points[3];
			fix16_t scale_outs[3];
			int indices[3];
			int ver=3;

			// order outputs:
			int32_t w_min = 0x7FFFFFFF;	
			int32_t w_max = 0;			
			int* shapes[3];
			for(int n=0; n<num_outputs; n++){
				shapes[n] = model_get_output_shape(model,n);
				w_min = MIN(shapes[n][3], w_min);
				w_max = MAX(shapes[n][3], w_max);
			}

			for(int i=0; i<num_outputs; i++){
				int o;	// proper order
				if(shapes[i][3]==w_min) o=0;		// stride min
				else if(shapes[i][3]==w_max) o=num_outputs-1;	// stride max
				else o=1;							// stride mid

				indices[o]=i;
				outputs[o] = (fix16_t*)(uintptr_t)o_buffers[i];
				outputs_int8[o] = (int8_t*)(uintptr_t)o_buffers[i];
				zero_points[o] = model_get_output_zeropoint(model,i);
				scale_outs[o]=model_get_output_scale_fix16_value(model,i);
			}

			//set masks
			int mask_0[] = {0,1,2};
			int mask_1[] = {3,4,5};
			int mask_2[] = {6,7,8};
			if(is_tiny==1){
				memcpy(mask_0,(int[]){1,2,3},3*sizeof(int));
			}
			
			int* masks[]= {mask_2, mask_1, mask_0}; //order should be: {[6,7,8],[3,4,5],[0,1,2]}
			int* tiny_masks[] = {mask_1, mask_0}; //order should be: {[3,4,5],[1,2,3]}
			
			fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
			fix16_t tiny_anchors[] = {F16(10),F16(14),F16(23),F16(27),F16(37),F16(58),F16(81),F16(82),F16(135),F16(169),F16(344),F16(319)}; // 2*num

			yolo_info_t cfg[num_outputs];
			for (int i = 0; i < num_outputs; i++) {				
				int* oshape = model_get_output_shape(model,indices[i]);
				yolo_info_t temp_cfg = {
					.version = ver,
					.input_dims = {in_dims[1], in_dims[2], in_dims[3]},
					.output_dims = {oshape[1], oshape[2], oshape[3]},
					.coords = coords,
					.classes = classes,
					.num = num,
					.anchors_length = anchors_length,
					.anchors = (is_tiny==1) ? tiny_anchors : anchors,
					.mask_length = mask_length,
					.mask = (is_tiny==1) ? tiny_masks[i] : masks[i],
				};		
				cfg[i] = temp_cfg;
			}
			if(int8_flag){
				valid_boxes = post_process_yolo_int8(outputs_int8, num_outputs, zero_points, scale_outs, cfg, thresh, iou, boxes, max_boxes);
			}
			else{
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}
				
		} else if (!strcmp(pptype, "YOLOV5")){ //ultralytics
			class_names = coco_classes;
			thresh =F16(.25);
			fix16_t *outputs[3];
			int8_t *outputs_int8[3];
			int zero_points[3];
			fix16_t scale_outs[3];
			int indices[3];
			int ver = 5;

			
			// order outputs:
			int32_t w_min = 0x7FFFFFFF;	
			int32_t w_max = 0;			
			int* shapes[3];
			for(int n=0; n<num_outputs; n++){
				shapes[n] = model_get_output_shape(model,n);
				w_min = MIN(shapes[n][3], w_min);
				w_max = MAX(shapes[n][3], w_max);
			}
			for(int i=0; i<num_outputs; i++){
				int o;	// proper order
				if(shapes[i][3]==w_min) o=0;		// stride min
				else if(shapes[i][3]==w_max) o=2;	// stride max
				else o=1;							// stride mid

				indices[o]=i;
				outputs[o] = (fix16_t*)(uintptr_t)o_buffers[i];
				outputs_int8[o] = (int8_t*)(uintptr_t)o_buffers[i];
				zero_points[o] = model_get_output_zeropoint(model,i);
				scale_outs[o]=model_get_output_scale_fix16_value(model,i);
			}

			fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
			int mask_0[] = {6,7,8};
			int mask_1[] = {3,4,5};
			int mask_2[] = {0,1,2};
						
			int* masks[] = {mask_0, mask_1, mask_2};
			
			yolo_info_t cfg[num_outputs];
			for (int i = 0; i < num_outputs; i++) {				
				int* oshape = model_get_output_shape(model,indices[i]);
				yolo_info_t temp_cfg = {
					.version = ver,
					.input_dims = {in_dims[1], in_dims[2], in_dims[3]},
					.output_dims = {oshape[1], oshape[2], oshape[3]},
					.coords = coords,
					.classes = classes,
					.num = num,
					.anchors_length = anchors_length,
					.anchors = anchors,
					.mask_length = mask_length,
					.mask = masks[i],
				};		
				cfg[i] = temp_cfg;
			}		
	
			if (int8_flag) {
				valid_boxes = post_process_yolo_int8(outputs_int8, num_outputs, zero_points, scale_outs, cfg, thresh, iou, boxes, max_boxes);
			} else {
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}
			

		} else if (!strcmp(pptype, "SSDV2")){
			fix16_t* output_buffers[12];
			char *is_vehicle = strstr(name, "ehicle");
			char *is_torch = strstr(name, "torch");
			if (is_vehicle == NULL) is_vehicle = strstr(name, "EHICLE");

			fix16_t confidence_threshold=F16(0.5);
			fix16_t nms_threshold=F16(0.4);
			int8_t* output_buffers_int8[12];
			fix16_t f16_scale[12];// = (fix16_t)model_get_output_scale_fix16_value(model,0); // get output scale
			int32_t zero_point[12];// = model_get_output_zeropoint(model,0); // get output zero
			if (is_torch) {
				for(int o=0;o<12;++o){
				    int* oshape = model_get_output_shape(model,o);
				    int idx;
				    if (oshape[2] == 1) {
					    idx = 5*2;
				    } else if (oshape[2] == 2) {
					    idx = 4*2;
				    } else if (oshape[2] == 3) {
					    idx = 3*2;
				    } else if (oshape[2] == 5) {
					    idx = 2*2;
				    } else if (oshape[2] == 10) {
					    idx = 1*2;
				    } else {
					    idx = 0*2;
				    }
				    if (oshape[1] != 24) {
					    idx += 1;
				    }
				    output_buffers[idx]=(fix16_t*)(uintptr_t)o_buffers[o];
				    output_buffers_int8[idx] = (int8_t*)(uintptr_t)o_buffers[o];
				    f16_scale[idx] = model_get_output_scale_fix16_value(model,o);
				    zero_point[idx] = model_get_output_zeropoint(model,o);
				}
				if(int8_flag){
				
					valid_boxes = post_process_ssd_torch_int8(boxes,max_boxes,output_buffers_int8,f16_scale,zero_point, 91,confidence_threshold,nms_threshold);
				} else{
					valid_boxes = post_process_ssd_torch(boxes,max_boxes,output_buffers,91,confidence_threshold,nms_threshold);
				}
				class_names = coco91_classes;
			} else if (is_vehicle) {
				for(int o=0;o<6;++o){
					output_buffers[2*o]=(fix16_t*)(uintptr_t)o_buffers[(6-1-o)*2];
					output_buffers[2*o+1]=(fix16_t*)(uintptr_t)o_buffers[(6-1-o)*2+1];
				}
				valid_boxes = post_process_vehicles(boxes,max_boxes,output_buffers,3,confidence_threshold,nms_threshold);
				class_names = vehicle_classes;
			} else {
				for(int o=0;o<12;++o){
					output_buffers[o]=(fix16_t*)(uintptr_t)o_buffers[(12-1-o)];
				}
				valid_boxes = post_process_ssdv2(boxes,max_boxes,output_buffers,91,confidence_threshold,nms_threshold);
				class_names = coco91_classes;
			}
		}

		char class_str[50];
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}

			if (class_names) { //class_names must be set, or prints the class id
				boxes[i].class_name = class_names[boxes[i].class_id];
				sprintf(class_str, "%s", boxes[i].class_name);
			} else {
				sprintf(class_str, "%d", boxes[i].class_id);
			}

#ifdef HARDWARE_DRAW
			int x = boxes[i].xmin,y=boxes[i].ymin;
			int w = boxes[i].xmax-boxes[i].xmin;
			int h = boxes[i].ymax-boxes[i].ymin;
			x = x*1920/input_w;
			w = w*1920/input_w;
			y = y*1080/input_h;
			h = h*1080/input_h;
			if(x<0 || y<0 || w<=0 || h<=0) {
				continue;
			}
			draw_box(x,y,w,h,5,get_colour_modulo(boxes[i].class_id),
					overlay_draw_frame,2048,1080);
			draw_label(boxes[i].class_name,x+5,y+5, overlay_draw_frame,2048,1080,WHITE);
#else
			printf("%s\t%.2f\t(%d, %d, %d, %d)\n",
					class_str,
					fix16_to_float(boxes[i].confidence),
					boxes[i].xmin,boxes[i].xmax,
					boxes[i].ymin,boxes[i].ymax);	
#endif
		}
	} else if (!strcmp(pptype, "POSENET")){
		const int MAX_TOTALPOSE=5;
		const int NUM_KEYPOINTS=17;
		poses_t r_poses[MAX_TOTALPOSE];
		
		int *output_dims = model_get_output_shape(model,1);
		int poseScoresH = output_dims[2]; 
		int poseScoresW = output_dims[3]; 	
		
	
		int outputStride = 16;
		int nmsRadius = 20;
		int pose_count = 0;
		fix16_t minPoseScore = F16(0.25);
		fix16_t scoreThreshold = F16(0.5);
		fix16_t score;
		if (int8_flag) {
			int zero_points[4];
			fix16_t scale_outs[4];
			int8_t* scores_8, *offsets_8, *displacementsFwd_8, *displacementsBwd_8;
			scores_8 = (int8_t*)(uintptr_t)o_buffers[1];
			offsets_8 = (int8_t*)(uintptr_t)o_buffers[0];
			displacementsFwd_8 = (int8_t*)(uintptr_t)o_buffers[2];
			displacementsBwd_8 = (int8_t*)(uintptr_t)o_buffers[3];
			for(int o=0; o<num_outputs; o++){
				zero_points[o] = model_get_output_zeropoint(model,o);
				scale_outs[o]=model_get_output_scale_fix16_value(model,o);
			}
			pose_count = decodeMultiplePoses_int8(r_poses,scores_8,offsets_8,displacementsFwd_8,displacementsBwd_8, outputStride, MAX_TOTALPOSE, scoreThreshold, nmsRadius, minPoseScore,poseScoresH,poseScoresW,zero_points,scale_outs); //actualpostprocess code
		}
		else{
			fix16_t* scores, *offsets, *displacementsFwd, *displacementsBwd;
			scores = (fix16_t*)(uintptr_t)o_buffers[1];
			offsets = (fix16_t*)(uintptr_t)o_buffers[0];
			displacementsFwd = (fix16_t*)(uintptr_t)o_buffers[2];
			displacementsBwd = (fix16_t*)(uintptr_t)o_buffers[3];
			pose_count = decodeMultiplePoses(r_poses,scores,offsets,displacementsFwd,displacementsBwd, outputStride, MAX_TOTALPOSE, scoreThreshold, nmsRadius, minPoseScore,poseScoresH,poseScoresW); //actualpostprocess code
		}
		
		
		
		int imageH, imageW;
		imageH = 273; //default img input dims
		imageW = 481; //default img input dims
		
		int *model_dims = model_get_input_shape(model,0);
		int modelInputH = model_dims[2];
		int modelInputW = model_dims[3];
		fix16_t scale_Y = fix16_div(fix16_from_int(imageH),fix16_from_int(modelInputH));
		fix16_t scale_X = fix16_div(fix16_from_int(imageW),fix16_from_int(modelInputW));

		
		//below scales up the image

		for(int i =0; i < pose_count; i++){
			for(int j=0;j <NUM_KEYPOINTS; j++){
				r_poses[i].keypoints[j][0] = fix16_mul(r_poses[i].keypoints[j][0],scale_Y);
				r_poses[i].keypoints[j][1] = fix16_mul(r_poses[i].keypoints[j][1],scale_X);
			}       

		}
		
		//print out points
		for(int i =0; i < pose_count; i++){
			printf("\nPose: %d\n",i);
			for(int j=0;j <NUM_KEYPOINTS; j++){
				int y = fix16_to_int(r_poses[i].keypoints[j][0]);
				int x = fix16_to_int(r_poses[i].keypoints[j][1]);
				score = r_poses[i].scores[j];
				printf("[%d , %d] %d \n",y,x, fix16_to_int(fix16_mul(score,F16(100))));	
			}
		}
	} else if (!strcmp(pptype, "ULTRALYTICS_POSE")){
		char **class_names = NULL;
		int valid_boxes = 0;
		int max_boxes = 100;
		const int boxes_len = 1024;
		fix16_box boxes[boxes_len];
		poses_t poses[boxes_len];
		const int NUM_KEYPOINTS=17;
		fix16_t thresh = F16(0.5);
		fix16_t iou = F16(0.5);

		int* outputs_shape[6+6];
		int8_t *outputs_int8[6+6];
		int zero_points[6+6];
		fix16_t scale_outs[6+6];
		
		int post_len;
		char class_str[50];

		// put outputs in this order
		// type:  {class_stride8, box_stride8,   class_stride16,  box_stride16,    class_stride32,  box_stride32}
		// shape: {[1,80,H/8,W/8],[1,64,H/8,W/8],[1,80,H/16,W/16],[1,64,H/16,W/16],[1,80,H/32,W/32],[1,64,H/32,W/32]}
		int32_t w_min = 0x7FFFFFFF;	// minimum width must be stride32
		int32_t w_max = 0;			// maximum width must be stride8
		int* shapes[num_outputs];
		int split = num_outputs == 12 ? 2:1; //is_pose flag being set by the split
		for(int n=0; n<num_outputs; n++){
			shapes[n] = model_get_output_shape(model,n);
			w_min = MIN(shapes[n][3], w_min);
			w_max = MAX(shapes[n][3], w_max);
		}
		for(int i=0; i<num_outputs; i++){
			int o;	// proper order
			if(shapes[i][3]==w_min) o=4;		// stride 8
			else if(shapes[i][3]==w_max) o=0;	// stride 32
			else o=2;							// stride 16
			if(shapes[i][1]==64) o+=1;			// box (otherwise class)
			if(shapes[i][1]==51) o = 6+o/2;			// box (otherwise class)
			if(shapes[i][1]==34) o = 6 +o;			// box (otherwise class)
			if(shapes[i][1]==17) o = 6 +o+1;			// box (otherwise class)
			outputs_shape[o] = shapes[i];
			outputs_int8[o] = (int8_t*)(uintptr_t)o_buffers[i];
			zero_points[o]=model_get_output_zeropoint(model,i);
			scale_outs[o]=model_get_output_scale_fix16_value(model,i);
		}	
		const int max_detections = 200;
		fix16_t post_buffer[max_detections*(4+1+17*3)];
		post_len = post_process_ultra_int8(outputs_int8, outputs_shape, post_buffer, thresh, zero_points, scale_outs, max_detections, 0, split,num_outputs);

		valid_boxes = post_process_ultra_nms(post_buffer, post_len, input_h, input_w, thresh, iou, boxes, poses, boxes_len, 1, 0, 1);

#ifdef HARDWARE_DRAW
		int radius = 6;
		int skeleton [19][2] = {{0,1} , {0,2}, {1,3}, {2,4}, {0,5}, {6,0}, {5,7}, {7,9}, {6,8}, {8,10}, {5,6}, {5,11}, {6,12}, {11,12}, {11,13}, {13,15}, {12,14}, {14,16}};
		int color;
		int imageH, imageW;
		imageH = 1080; //default img feed dims
		imageW = 1920; //default img feed dims
		fix16_t kp_thresh = F16(0.9);
		int *model_dims = model_get_input_shape(model,0);
		int modelInputH = model_dims[2];
		int modelInputW = model_dims[3];
		fix16_t scale_Y = fix16_div(fix16_from_int(imageH),fix16_from_int(modelInputH));
		fix16_t scale_X = fix16_div(fix16_from_int(imageW),fix16_from_int(modelInputW));
		
		//scales up the image
		for(int i = 0; i < valid_boxes; i++) {
			for(int j = 0; j < NUM_KEYPOINTS; j++) {
				poses[i].keypoints[j][1] = fix16_mul(poses[i].keypoints[j][1],scale_Y);
				poses[i].keypoints[j][0] = fix16_mul(poses[i].keypoints[j][0],scale_X);
			}       
		}
		
		//draw boxes
		for(int i =0; i < valid_boxes; i++) {
			if(boxes[i].confidence == 0){
				continue;
			}
			if (class_names) { //class_names must be set, or prints the class id
				boxes[i].class_name = class_names[boxes[i].class_id];
				sprintf(class_str, "%s", boxes[i].class_name);
			} else {
				sprintf(class_str, "%d", boxes[i].class_id);
			}
//draw box here
			int x = (boxes[i].xmin+boxes[i].xmax)/2;
			int y = (boxes[i].ymin+boxes[i].ymax)/2;
			int w = boxes[i].xmax-boxes[i].xmin;
			int h = boxes[i].ymax-boxes[i].ymin;
			x = boxes[i].xmin;
			y = boxes[i].ymin;
			
			x = x*1920/input_w;
			w = w*1920/input_w;
			y = y*1080/input_h;
			h = h*1080/input_h;
			draw_box(x,y,w,h,5,get_colour_modulo(boxes[i].class_id),
					overlay_draw_frame,2048,1080);
			
			//Draw points and lines		
			for(int z=0; z<18; z++) { 
				int y_source = fix16_to_int(poses[i].keypoints[skeleton[z][0]][1]); //source y
				int x_source = fix16_to_int(poses[i].keypoints[skeleton[z][0]][0]); //source x
				int y_target = fix16_to_int(poses[i].keypoints[skeleton[z][1]][1]); //target y
				int x_target = fix16_to_int(poses[i].keypoints[skeleton[z][1]][0]); //target y
				
				if(poses[i].scores[skeleton[z][0]]>kp_thresh && poses[i].scores[skeleton[z][1]]>kp_thresh) {
					if(skeleton[z][1]%2==1){
						color = GET_COLOUR(147, 20, 255, 255);
					}	else {
						color = GET_COLOUR(0, 255, 255, 255);						
					}
					draw_box(x_target,y_target,3,3,radius, color, overlay_draw_frame,2048,1080); //draw points
					if((skeleton[z][0])%2==1 && (skeleton[z][1])%2==0){
						color = GET_COLOUR(255, 255, 0, 255);
					}
					draw_line(x_source, y_source, x_target, y_target,color, overlay_draw_frame, 2048,1080, 2,0); //connect edges
				}
			}
		
		}

#else		
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}

			if (class_names) { //class_names must be set, or prints the class id
				boxes[i].class_name = class_names[boxes[i].class_id];
				sprintf(class_str, "%s", boxes[i].class_name);
			} else {
				sprintf(class_str, "%d", boxes[i].class_id);
			}

			printf("(%d, %d, %d, %d) %d\n",
					(boxes[i].xmin+boxes[i].xmax)/2,
					(boxes[i].ymin+boxes[i].ymax)/2,
					boxes[i].xmax-boxes[i].xmin,
					boxes[i].ymax-boxes[i].ymin,
					fix16_to_int(fix16_mul(boxes[i].confidence, F16(100))));
			//print out points
			for(int j=0;j <NUM_KEYPOINTS; j++){
				int x = fix16_to_int(poses[i].keypoints[j][0]);
				int y = fix16_to_int(poses[i].keypoints[j][1]);
				printf("\t(%d , %d) %d\n",x,y, fix16_to_int(fix16_mul(poses[i].scores[j],F16(100))));	
			}
		}
#endif		

	} else if (!strcmp(pptype, "ULTRALYTICS_OBB")){
		char **class_names = NULL;
		int valid_boxes = 0;
		int max_boxes = 2000;
		const int boxes_len = 2048;
		fix16_box boxes[boxes_len];
		fix16_t thresh = F16(0.85);
		fix16_t iou = F16(0.5);

		class_names = dota_classes;
		int* outputs_shape[6+3];
		int8_t *outputs_int8[6+3];
		int zero_points[6+3];
		fix16_t scale_outs[6+3];

		// put outputs in this order
		// type:  {class_stride8, box_stride8,   class_stride16,  box_stride16,    class_stride32,  box_stride32}
		// shape: {[1,80,H/8,W/8],[1,64,H/8,W/8],[1,80,H/16,W/16],[1,64,H/16,W/16],[1,80,H/32,W/32],[1,64,H/32,W/32]}
		int32_t w_min = 0x7FFFFFFF;	// minimum width must be stride32
		int32_t w_max = 0;			// maximum width must be stride8
		int* shapes[6+3];
		for(int n=0; n<6+3; n++){
			shapes[n] = model_get_output_shape(model,n);
			w_min = MIN(shapes[n][3], w_min);
			w_max = MAX(shapes[n][3], w_max);
		}
		for(int i=0; i<6+3; i++){
			int o;	// proper order
			if(shapes[i][3]==w_min) o=4;		// stride 8
			else if(shapes[i][3]==w_max) o=0;	// stride 32
			else o=2;							// stride 16
			if(shapes[i][1]==64) o+=1;			// box (otherwise class)
			if(shapes[i][1]==1) o=6+o/2;			// angles
			outputs_shape[o] = shapes[i];
			outputs_int8[o] = (int8_t*)(uintptr_t)o_buffers[i];
			zero_points[o]=model_get_output_zeropoint(model,i);
			scale_outs[o]=model_get_output_scale_fix16_value(model,i);
		}
		const int max_detections = 4000;
		fix16_t post_buffer[max_detections*4+15+1];
		int post_len;
		post_len = post_process_ultra_int8(outputs_int8, outputs_shape, post_buffer, thresh, zero_points, scale_outs, max_detections, 1, 0, num_outputs);
		valid_boxes = post_process_ultra_nms(post_buffer, post_len, input_h, input_w, thresh, iou, boxes, NULL, boxes_len, 15, 1, 0);
		char class_str[50];
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}

			if (class_names) { //class_names must be set, or prints the class id
				boxes[i].class_name = class_names[boxes[i].class_id];
				sprintf(class_str, "%s", boxes[i].class_name);
			} else {
				sprintf(class_str, "%d", boxes[i].class_id);
			}

			printf("%s\t%3.2f\t(%d, %d, %d, %d) %3.2f\n",
					class_str,
					fix16_to_float(boxes[i].confidence),
					boxes[i].x,boxes[i].y,
					boxes[i].w,boxes[i].h,
					fix16_to_float(boxes[i].angle));
		}
	}  else {
#ifdef HARDWARE_DRAW
#else		
		printf("Unknown post processing type %s, skipping post process\n", pptype);	
#endif	
		return -1;
	}
	return 0;
}